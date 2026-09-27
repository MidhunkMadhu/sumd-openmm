"""
parallel.py

Rank layout for parallel = mpi. Pure python (mpi4py is imported only by the
caller), so the GPU assignment rules are unit-testable anywhere.

mpi_mode = multi_node
    One MPI rank per node, one GPU each. A second rank on the same node is an
    error: it would share the GPU and halve both walkers' speed.
    gpu_devices = auto  -> OpenMM's default device (the first visible GPU,
                           respecting CUDA_VISIBLE_DEVICES)
    gpu_devices = 2     -> device 2 on every node

mpi_mode = multi_gpu
    Several ranks per node, one GPU per rank, chosen by the NODE-LOCAL rank.
    Also covers several nodes with several GPUs each.
    gpu_devices = auto        -> local rank k uses device k; if the scheduler
                                 already binds one GPU per rank
                                 (CUDA_VISIBLE_DEVICES, or on AMD GPUs
                                 ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES,
                                 holds one id), device 0
    gpu_devices = 0,1,2,3     -> local rank k uses the k-th listed device
"""


class LayoutError(RuntimeError):
    pass


def parse_devices(value):
    """'auto' -> None; '0,1, 3' -> [0, 1, 3]."""
    if value is None:
        return None

    v = str(value).strip().lower()
    if v in ("", "auto", "none"):
        return None

    return [int(x) for x in v.replace(",", " ").split()]


VISIBILITY_VARIABLES = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES")


def visible_devices(env):
    """
    GPUs the scheduler exposes to this process, or None if unrestricted.
    NVIDIA uses CUDA_VISIBLE_DEVICES; AMD ROCm uses ROCR_VISIBLE_DEVICES
    (Slurm's --gpu-bind) and HIP_VISIBLE_DEVICES. The most restrictive set wins.
    """
    found = []
    for name in VISIBILITY_VARIABLES:
        v = env.get(name)
        if v is not None and v.strip() != "":
            found.append([x for x in v.split(",") if x.strip() != ""])
    return min(found, key=len) if found else None


def assign_device(mode, gpu_devices, local_rank, local_size, env, host="this node"):
    """
    Device index (string, as OpenMM's DeviceIndex property wants) for one
    rank, or None to let OpenMM choose. Raises LayoutError on an impossible
    layout, with a message saying what to change.
    """
    devices = parse_devices(gpu_devices)
    vis = visible_devices(env)

    if mode == "multi_node":
        if local_size > 1:
            raise LayoutError(
                "mpi_mode = multi_node expects one MPI rank per node, but %d ranks share %s. "
                "Launch one rank per node (e.g. srun --ntasks-per-node=1), or use "
                "mpi_mode = multi_gpu if the node has one GPU per rank." % (local_size, host))
        return None if devices is None else str(devices[0])

    if mode != "multi_gpu":
        raise LayoutError("unknown mpi_mode %r" % mode)

    if devices is not None:
        if local_size > len(devices):
            raise LayoutError(
                "%d ranks on %s but gpu_devices lists only %d devices (%s)"
                % (local_size, host, len(devices), ",".join(map(str, devices))))
        return str(devices[local_rank])

    if vis is not None and len(vis) == 1:
        return "0"          # scheduler bound exactly one GPU to this rank

    if vis is not None and local_size > len(vis):
        raise LayoutError(
            "%d ranks on %s but the GPU visibility variables expose %d GPUs (%s)"
            % (local_size, host, len(vis), ",".join(vis)))

    return str(local_rank)


def check_unique_devices(layout):
    """
    layout: list of dicts with host, device. Two ranks on one host with the
    same explicit device would share a GPU. Returns a list of warnings.
    (Ranks with an individually bound GPU all report device 0, so only
    ranks whose CUDA_VISIBLE_DEVICES is identical are compared.)
    """
    seen, warnings = {}, []

    for r in layout:
        if r.get("device") is None:
            continue
        key = (r["host"], r.get("cuda_visible"), r["device"])
        if key in seen:
            warnings.append("ranks %d and %d both use device %s on %s"
                            % (seen[key], r["rank"], r["device"], r["host"]))
        else:
            seen[key] = r["rank"]

    return warnings


def walker_ranks(n_walkers, size, first_rank=0):
    """
    Rank for every walker, round-robin, starting at `first_rank`: the rank
    whose Context already holds the parent gets walker 0, so it can continue
    in place instead of restoring.
    """
    order = [first_rank] + [r for r in range(size) if r != first_rank]
    return [order[w % size] for w in range(n_walkers)]
