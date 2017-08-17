using Juno: info
using .Batches: tobatch

"""
  @cb for ... end t expr

Run the for loop, executing `expr` every `t` seconds.
"""
macro cb(ex, t, f)
  @assert isexpr(ex, :for)
  cond, body = ex.args
  @esc t f cond body
  :(let
    t0 = time_ns()
    dt = $t*1e9
    @progress $(Expr(:for, cond, quote
      t = time_ns()
      if t - t0 > dt
        t0 = t
        f = () -> $f
        f()
      end
      $body
    end))
  end)
end

"""
Returns a function that when invoked, will only be triggered at most once
during `timeout` seconds. Normally, the throttled function will run
as much as it can, without ever going more than once per `wait` duration;
but if you'd like to disable the execution on the leading edge, pass
`leading=false`. To enable execution on the trailing edge, ditto.
"""
function throttle(f, timeout; leading=true, trailing=false)
  cooldown = true
  later = nothing

  function throttled(args...; kwargs...)
    yield()

    if cooldown
      if leading
        f(args...; kwargs...)
      else
        later = () -> f(args...; kwargs...)
      end

      cooldown = false
      @schedule try
        while (sleep(timeout); later != nothing)
          later()
          later = nothing
        end
      finally
        cooldown = true
      end
    elseif trailing
      later = () -> f(args...; kwargs...)
    end

    nothing
  end
end

function train!(m, train; cb = [],
                epoch = 1, η = 0.1, loss = mse)
    callback = throttle(()->foreach(f -> f(), cb), 5)

    @progress for e in 1:epoch
      info("Epoch $e")
      for (x, y) in train
        x, y = mapt(tobatch, (x, y))
        ŷ = m(x)
        any(isnan, ŷ) && error("NaN")
        Δ = back!(loss, 1, ŷ, y)
        back!(m, Δ, x)
        update!(m, η)
        callback()
      end
    end
    return m
end
