# Copyright (c) 2020 CRS4
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pytest

import pyecvl._core.ecvl as ecvl_core
import pyecvl.ecvl as ecvl_py


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugmentationParam(ecvl):
    ap = ecvl.AugmentationParam()
    min_, max_ = 0.1, 1.2
    ap = ecvl.AugmentationParam(min_, max_)
    assert ap.min_ == pytest.approx(min_)
    assert ap.max_ == pytest.approx(max_)
    ecvl.AugmentationParam.SetSeed(12345)
    ap.GenerateValue()
    assert min_ <= ap.value_ <= max_


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugRotate(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugRotate([30, 50])
    a.Apply(img)
    a = ecvl.AugRotate([30, 50], [2, 3])
    a.Apply(img)
    a = ecvl.AugRotate([30, 50], [2, 3], 1.1)
    a.Apply(img)
    a = ecvl.AugRotate([30, 50], [2, 3], 1.1, ecvl.InterpolationType.nearest)
    a.Apply(img)
    # fromtext
    f = ecvl.AugRotate if ecvl is ecvl_core else ecvl.AugRotate.fromtext
    a = f('angle=[30, 50] center=(2, 3) scale=1.1 interp="nearest"\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugResizeDim(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugResizeDim([4, 3])
    a.Apply(img)
    a = ecvl.AugResizeDim([4, 3], ecvl.InterpolationType.nearest)
    a.Apply(img)
    # fromtext
    f = ecvl.AugResizeDim if ecvl is ecvl_core else ecvl.AugResizeDim.fromtext
    a = f('dims=(4, 3) interp="linear"\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugResizeScale(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugResizeScale([0.5, 0.5])
    a.Apply(img)
    a = ecvl.AugResizeScale([0.5, 0.5], ecvl.InterpolationType.nearest)
    a.Apply(img)
    # fromtext
    f = ecvl.AugResizeScale if ecvl is ecvl_core else \
        ecvl.AugResizeScale.fromtext
    a = f('scale=(0.5, 0.5) interp="linear"\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugFlip(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugFlip(0.5)
    a.Apply(img)
    # fromtext
    f = ecvl.AugFlip if ecvl is ecvl_core else ecvl.AugFlip.fromtext
    a = f('p=0.5\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugMirror(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugMirror(0.5)
    a.Apply(img)
    # fromtext
    f = ecvl.AugMirror if ecvl is ecvl_core else ecvl.AugMirror.fromtext
    a = f('p=0.5\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugGaussianBlur(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugGaussianBlur([0.2, 0.4])
    a.Apply(img)
    # fromtext
    f = ecvl.AugGaussianBlur if ecvl is ecvl_core else \
        ecvl.AugGaussianBlur.fromtext
    a = f('sigma=[0.2, 0.4]\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugAdditiveLaplaceNoise(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugAdditiveLaplaceNoise([255 * 0.05, 255 * 0.09])
    a.Apply(img)
    # fromtext
    f = ecvl.AugAdditiveLaplaceNoise if ecvl is ecvl_core else \
        ecvl.AugAdditiveLaplaceNoise.fromtext
    a = f('std_dev=[12.5, 23.1]\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugAdditivePoissonNoise(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugAdditivePoissonNoise([2.0, 3.0])
    a.Apply(img)
    # fromtext
    f = ecvl.AugAdditivePoissonNoise if ecvl is ecvl_core else \
        ecvl.AugAdditivePoissonNoise.fromtext
    a = f('lambda=[2.0, 3.0]\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugGammaContrast(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugGammaContrast([3, 4])
    a.Apply(img)
    # fromtext
    f = ecvl.AugGammaContrast if ecvl is ecvl_core else \
        ecvl.AugGammaContrast.fromtext
    a = f('gamma=[3, 4]\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugCoarseDropout(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    a = ecvl.AugCoarseDropout([0.5, 0.7], [0.1, 0.2], 0.4)
    a.Apply(img)
    # fromtext
    f = ecvl.AugCoarseDropout if ecvl is ecvl_core else \
        ecvl.AugCoarseDropout.fromtext
    a = f('p=[0.5, 0.7] drop_size=[0.1, 0.2] per_channel=0.4\n')
    a.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_AugmentationFactory(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    # one arg
    a = ecvl.AugmentationFactory.create('AugFlip p=0.5\n')
    a.Apply(img)
    # two args
    a = ecvl.AugmentationFactory.create('AugFlip', 'p=0.5\n')
    a.Apply(img)
    # container
    txt = ('SequentialAugmentationContainer\n'
           'AugRotate angle=[-5,5] center=(0,0) scale=0.5 interp="linear"\n'
           'AugAdditiveLaplaceNoise std_dev=[0,0.51]\n'
           'AugCoarseDropout p=[0,0.55] drop_size=[0.02,0.1] per_channel=0\n'
           'AugAdditivePoissonNoise lambda=[0,40]\n'
           'AugResizeDim dims=(30,30) interp="linear"\n'
           'end\n')
    c = ecvl.AugmentationFactory.create(txt)
    c.Apply(img)


@pytest.mark.parametrize("ecvl", [ecvl_core, ecvl_py])
def test_SequentialAugmentationContainer(ecvl):
    img = ecvl.Image([5, 4, 3], ecvl.DataType.uint8, "xyc", ecvl.ColorType.BGR)
    # from list
    c = ecvl.SequentialAugmentationContainer([
        ecvl.AugRotate([-5, 5]),
        ecvl.AugMirror(.5),
    ])
    c.Apply(img)
    # fromtext
    txt = ('AugFlip p=0.2\n'
           'AugMirror p=0.2\n'
           'end\n')
    f = ecvl.SequentialAugmentationContainer if ecvl is ecvl_core else \
        ecvl.SequentialAugmentationContainer.fromtext
    c = f(txt)
    c.Apply(img)
