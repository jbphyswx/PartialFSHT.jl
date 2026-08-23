#=
Run every example in order and report which succeeded.

Each example asserts the property it demonstrates, so a nonzero exit here means the package changed
behaviour the manual describes — not merely that a number moved.
=#

using Printf: Printf

const HERE = @__DIR__
const FILES = sort(filter(f -> occursin(r"^\d\d_.*\.jl$", f), readdir(HERE)))

failed = String[]
for f in FILES
    println("\n" * "="^78)
    println("  $f")
    println("="^78)
    t = @elapsed try
        include(joinpath(HERE, f))
    catch err
        push!(failed, f)
        showerror(stdout, err)
        println()
    end
    Printf.@printf("[%s] %s in %.1f s\n", f in failed ? "FAIL" : " ok ", f, t)
end

println("\n" * "="^78)
Printf.@printf("%d of %d examples ran clean\n", length(FILES) - length(failed), length(FILES))
isempty(failed) || (Printf.@printf("failed: %s\n", join(failed, ", ")); exit(1))
