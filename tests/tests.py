import research_toolbox.tb_utils as tb_ut
import research_toolbox.tb_remote as tb_re

def test_zip_toggle():
    xs = [[1, 2, 3], [2, 3, 4]]
    out_xs = tb_ut.zip_toggle(tb_ut.zip_toggle(xs))
    convert_to_tuple_fn = lambda ys: tuple(map(tuple, ys))
    assert convert_to_tuple_fn(xs) == convert_to_tuple_fn(out_xs)
