module CUDAint

using ..CUDA

import ..Flux: Flux, Chain
using ChainRulesCore
import NNlib, NNlibCUDA

include("cudnn.jl")

# This is NOT safe within jacobian. It needs something like https://github.com/FluxML/Zygote.jl/pull/1340
# so that Zygote can tell use when it is safe.
function ChainRulesCore.rrule(cfg::RuleConfig, ch::Chain, x::CuArray)
# function ChainRulesCore.rrule(cfg::ZygoteRuleConfig{Context{false,true}}, ch::Chain, x::AbstractArray)
  duo = accumulate(ch.layers; init=(x, nothing)) do (input, _), layer
    out, back = rrule_via_ad(cfg, layer, input)
  end
  outs = map(first, duo)
  backs = map(last, duo)
  function un_chain(dout)
    multi = accumulate(reverse(backs); init=(nothing, dout)) do (_, delta), back
      dlayer, din = back(delta)
    end
    layergrads = reverse(map(unthunk∘first, multi))
    xgrad = unthunk(last(multi[end]))
    foreach(CUDA.unsafe_free!, outs)
    foreach(CUDA.unsafe_free!, map(last, multi[1:end-1]))
    # foreach(maybe_final, outs)  # using the Zygote PR
    # foreach(maybe_final, map(last, multi[1:end-1]))
    return (Tangent{Chain}(; layers = layergrads), xgrad)
  end
  outs[end], un_chain
end

# For testing we write NaN into non-CuArrays (piratically):
CUDA.unsafe_free!(x::Array{<:AbstractFloat}) = fill!(x, NaN)
CUDA.unsafe_free!(x::Flux.Zygote.Fill) = nothing
CUDA.unsafe_free!(x::Array) = nothing

end  # module CUDAint
