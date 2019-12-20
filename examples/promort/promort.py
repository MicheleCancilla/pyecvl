"""\
PROMORT example.
"""

import argparse
import random
import sys

import pyecvl._core.ecvl as ecvl
import pyeddl._core.eddl as eddl
import pyeddl._core.eddlT as eddlT


def VGG16(in_layer, num_classes):
    x = in_layer
    x = eddl.ReLu(eddl.Conv(x, 64, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 64, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 128, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 128, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 256, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 256, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.ReLu(eddl.Conv(x, 512, [3, 3]))
    x = eddl.MaxPool(eddl.ReLu(eddl.Conv(x, 512, [3, 3])), [2, 2], [2, 2])
    x = eddl.Reshape(x, [-1])
    x = eddl.ReLu(eddl.Dense(x, 256))
    x = eddl.Softmax(eddl.Dense(x, num_classes))
    return x


def main(args):
    num_classes = 2
    size = [256, 256]  # size of images

    in_ = eddl.Input([3, size[0], size[1]])
    out = VGG16(in_, num_classes)
    net = eddl.Model([in_], [out])
    eddl.build(
        net,
        eddl.rmsprop(1e-5),
        ["soft_cross_entropy"],
        ["categorical_accuracy"],
        eddl.CS_GPU([1]) if args.gpu else eddl.CS_CPU()
    )
    eddl.summary(net)

    print("Reading dataset")
    d = ecvl.DLDataset(args.in_ds, args.batch_size, "training")
    x_train = eddlT.create([args.batch_size, d.n_channels_, size[0], size[1]])
    y_train = eddlT.create([args.batch_size, len(d.classes_)])
    num_samples = len(d.GetSplit())
    num_batches = num_samples // args.batch_size
    indices = list(range(args.batch_size))

    for i in range(args.epochs):
        eddl.reset_loss(net)
        s = d.GetSplit()
        random.shuffle(s)
        d.split_.training_ = s
        d.current_batch_ = 0
        for j in range(num_batches):
            print("Epoch %d/%d (batch %d/%d) - " %
                  (i + 1, args.epochs, j + 1, num_batches), end="", flush=True)
            ecvl.LoadBatch(d, size, x_train, y_train)
            x_train.div_(255.0)
            tx, ty = [x_train], [y_train]
            eddl.train_batch(net, tx, ty, indices)
            eddl.print_loss(net, j)
            print()
            d.current_batch_ += 1

    eddl.save(net, "promort_checkpoint.bin", "bin")

    print("Evaluation")
    d.SetSplit("validation")
    num_samples = len(d.GetSplit())
    num_batches = num_samples // args.batch_size
    d.current_batch_ = 0
    for i in range(num_batches):
        print("batch %d / %d - " % (i, num_batches), end="", flush=True)
        ecvl.LoadBatch(d, size, x_train, y_train)
        x_train.div_(255.0)
        eddl.evaluate(net, [x_train], [y_train])
        d.current_batch_ += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("in_ds", metavar="INPUT_DATASET")
    parser.add_argument("--epochs", type=int, metavar="INT", default=10)
    parser.add_argument("--batch-size", type=int, metavar="INT", default=8)
    parser.add_argument("--gpu", action="store_true")
    main(parser.parse_args(sys.argv[1:]))
