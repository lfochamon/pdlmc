import jax
import jax.lax as lax
import jax.random as jr
from jax import jit, tree_map
from jax.experimental import host_callback


def random_split_like_tree(rng_key, target):
    treedef = jax.tree_structure(target)
    keys = jax.random.split(rng_key, treedef.num_leaves)
    return jax.tree_unflatten(treedef, keys)


def tree_random_normal_like(rng_key, target):
    out_key, rng_key = jr.split(rng_key, 2)
    keys_tree = random_split_like_tree(rng_key, target)
    return out_key, tree_map(
        lambda l, k: jax.random.normal(k, l.shape, l.dtype),
        target,
        keys_tree,
    )


def _print_consumer(arg, transform):
    iter_num, num_samples = arg
    print(f"Iter {iter_num:,} / {num_samples:,}")


@jit
def progbar(arg, result):
    iter_num, num_samples, print_rate = arg
    result = lax.cond(
        iter_num % print_rate == 0,
        lambda _: host_callback.id_tap(_print_consumer, (iter_num, num_samples), result=result),
        lambda _: result,
        operand=None,
    )
    return result


def progress_bar_scan(num_samples):
    """
    Decorator that adds a progress bar to `body_fun` used in `lax.scan`.
    Note that `body_fun` must be looping over `jnp.arange(num_samples)`.
    This means that `iter_num` is the current iteration number
    """

    def _progress_bar_scan(func):
        print_rate = int(num_samples / 10)

        def wrapper_progress_bar(carry, iter_num):
            iter_num = progbar((iter_num + 1, num_samples, print_rate), iter_num)
            return func(carry, iter_num)

        return wrapper_progress_bar

    return _progress_bar_scan
