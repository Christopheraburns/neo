import mxnet as mx
import numpy as np




img = mx.image.imread('1.jpg')# convert into format (batch, RGB, width, height)
img = mx.image.imresize(img, 512, 512) # resize
img = img.transpose((2, 0, 1)) # Channel first
img = img.expand_dims(axis=0) # batchify
img = img.astype(dtype='float32')

sym, args, aux = mx.model.load_checkpoint('compiled', 0)
args['data'] = img

# Inferentia context
ctx = mx.neuron()

exe = sym.bind(ctx=ctx, args=args, aux_states=aux, grad_req='null')

exe.forward(data=img)

