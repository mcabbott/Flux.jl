# Utility Functions

Flux contains some utility functions for working with data; these functions
help create inputs for your models or batch your dataset.
Other functions can be used to initialize your layers or to regularly execute
callback functions.

## Working with Data

```@docs
Flux.unsqueeze
Flux.stack
Flux.unstack
Flux.chunk
Flux.frequencies
Flux.batch
Flux.batchseq
Base.rpad(v::AbstractVector, n::Integer, p)
```

## [Layer Initialisation](@id man-init)

By default Flux initialises the weights of convolutional layers and recurrent cells with `glorot_uniform`.
To change this, you can give a function to the `init` keyword. For example:

```jldoctest; setup = :(using Flux)
julia> conv = Conv((3, 3), 1 => 8, relu; init=Flux.glorot_normal)
Conv((3, 3), 1=>8, relu)
```

To perform more complicated initialisation, it's recommended to write a function which creates the layer.
For example, this should match Pytorch's nn.Linear layer defaults:

```jldoctest; setup = :(using Flux)
julia> function pydense(in, out, σ=identity; bias=true)
         W = Flux.kaiming_uniform(out, in, gain=sqrt(2/5))
         fan_in, _ = Flux.nfan(out, in)
         b = (rand(out) .- 1/2) .* 2 ./ sqrt(fan_in) .|> Float32
         Dense(W, bias && b, σ)
       end;

julia> pydense(28^2, 32, tanh, bias=false)
Dense(784, 32, tanh; bias=false)
```

```@docs
Flux.glorot_uniform
Flux.glorot_normal
Flux.kaiming_uniform
Flux.kaiming_normal
Flux.orthogonal
Flux.sparse_init
Flux.nfan
```

## Model Building

Flux provides some utility functions to help you generate models in an automated fashion.

[`outputsize`](@ref) enables you to calculate the output sizes of layers like [`Conv`](@ref)
when applied to input samples of a given size. This is achieved by passing a "dummy" array into
the model that preserves size information without running any computation.
`outputsize(f, inputsize)` works for all layers (including custom layers) out of the box.
By default, `inputsize` expects the batch dimension,
but you can exclude the batch size with `outputsize(f, inputsize; padbatch=true)` (assuming it to be one).

Using this utility function lets you automate model building for various inputs like so:
```julia
"""
    make_model(width, height, inchannels, nclasses;
               layer_config = [16, 16, 32, 32, 64, 64])

Create a CNN for a given set of configuration parameters.

# Arguments
- `width`: the input image width
- `height`: the input image height
- `inchannels`: the number of channels in the input image
- `nclasses`: the number of output classes
- `layer_config`: a vector of the number of filters per each conv layer
"""
function make_model(width, height, inchannels, nclasses;
                    layer_config = [16, 16, 32, 32, 64, 64])
  # construct a vector of conv layers programmatically
  conv_layers = [Conv((3, 3), inchannels => layer_config[1])]
  for (infilters, outfilters) in zip(layer_config, layer_config[2:end])
    push!(conv_layers, Conv((3, 3), infilters => outfilters))
  end

  # compute the output dimensions for the conv layers
  # use padbatch=true to set the batch dimension to 1
  conv_outsize = Flux.outputsize(conv_layers, (width, height, nchannels); padbatch=true)

  # the input dimension to Dense is programatically calculated from
  #  width, height, and nchannels
  return Chain(conv_layers..., Dense(prod(conv_outsize), nclasses))
end
```

```@docs
Flux.outputsize
```

## Model Abstraction

```@docs
Flux.destructure
Flux.nfan
```

## Callback Helpers

```@docs
Flux.throttle
Flux.stop
Flux.skip
```
