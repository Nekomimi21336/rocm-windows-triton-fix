module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @triton_per_fused__to_copy__unsafe_view_add_clone_div_expand_fill_mul_pow_sigmoid_silu_sub_sum_transpose_view_1(%arg0: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg4: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg5: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg6: !tt.ptr<bf16> {tt.divisibility = 16 : i32}, %arg7: i32 {tt.divisibility = 16 : i32}, %arg8: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c393216_i32 = arith.constant 393216 : i32
    %c32768_i32 = arith.constant 32768 : i32
    %c64_i32 = arith.constant 64 : i32
    %c6144_i32 = arith.constant 6144 : i32
    %c512_i32 = arith.constant 512 : i32
    %c12_i32 = arith.constant 12 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_0 = arith.constant dense<2.000000e+00> : tensor<1x64xf32, #blocked>
    %cst_1 = arith.constant dense<1.562500e-02> : tensor<1x1xf32, #blocked1>
    %cst_2 = arith.constant dense<-5.000000e-01> : tensor<1x1xf32, #blocked1>
    %cst_3 = arith.constant dense<1.000000e+00> : tensor<1x64xf32, #blocked>
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #blocked2>
    %2 = ttg.convert_layout %1 : tensor<64xi32, #blocked2> -> tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>>
    %3 = tt.expand_dims %2 {axis = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 0, parent = #blocked3}>> -> tensor<1x64xi32, #blocked3>
    %4 = ttg.convert_layout %3 : tensor<1x64xi32, #blocked3> -> tensor<1x64xi32, #blocked>
    %5 = arith.remsi %0, %c12_i32 : i32
    %6 = arith.divsi %0, %c12_i32 : i32
    %7 = arith.remsi %6, %c512_i32 : i32
    %8 = arith.divsi %0, %c6144_i32 : i32
    %9 = arith.muli %0, %c64_i32 : i32
    %10 = tt.splat %9 : i32 -> tensor<1x64xi32, #blocked>
    %11 = arith.addi %4, %10 : tensor<1x64xi32, #blocked>
    %12 = tt.splat %arg0 : !tt.ptr<bf16> -> tensor<1x64x!tt.ptr<bf16>, #blocked>
    %13 = tt.addptr %12, %11 : tensor<1x64x!tt.ptr<bf16>, #blocked>, tensor<1x64xi32, #blocked>
    %14 = tt.load %13 : tensor<1x64x!tt.ptr<bf16>, #blocked>
    %15 = arith.extf %14 : tensor<1x64xbf16, #blocked> to tensor<1x64xf32, #blocked>
    %16 = tt.splat %arg1 : !tt.ptr<bf16> -> tensor<1x64x!tt.ptr<bf16>, #blocked>
    %17 = tt.addptr %16, %11 : tensor<1x64x!tt.ptr<bf16>, #blocked>, tensor<1x64xi32, #blocked>
    %18 = tt.load %17 : tensor<1x64x!tt.ptr<bf16>, #blocked>
    %19 = arith.extf %18 : tensor<1x64xbf16, #blocked> to tensor<1x64xf32, #blocked>
    %20 = arith.muli %5, %c64_i32 : i32
    %21 = tt.splat %20 : i32 -> tensor<1x64xi32, #blocked>
    %22 = arith.addi %4, %21 : tensor<1x64xi32, #blocked>
    %23 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1x64x!tt.ptr<f32>, #blocked>
    %24 = tt.addptr %23, %22 : tensor<1x64x!tt.ptr<f32>, #blocked>, tensor<1x64xi32, #blocked>
    %25 = tt.load %24 evictionPolicy = evict_last : tensor<1x64x!tt.ptr<f32>, #blocked>
    %26 = arith.muli %7, %c64_i32 : i32
    %27 = tt.splat %26 : i32 -> tensor<1x64xi32, #blocked>
    %28 = arith.addi %4, %27 : tensor<1x64xi32, #blocked>
    %29 = arith.muli %5, %c32768_i32 : i32
    %30 = tt.splat %29 : i32 -> tensor<1x64xi32, #blocked>
    %31 = arith.addi %28, %30 : tensor<1x64xi32, #blocked>
    %32 = arith.muli %8, %c393216_i32 : i32
    %33 = tt.splat %32 : i32 -> tensor<1x64xi32, #blocked>
    %34 = arith.addi %31, %33 : tensor<1x64xi32, #blocked>
    %35 = tt.splat %arg3 : !tt.ptr<bf16> -> tensor<1x64x!tt.ptr<bf16>, #blocked>
    %36 = tt.addptr %35, %34 : tensor<1x64x!tt.ptr<bf16>, #blocked>, tensor<1x64xi32, #blocked>
    %37 = tt.load %36 : tensor<1x64x!tt.ptr<bf16>, #blocked>
    %38 = arith.extf %37 : tensor<1x64xbf16, #blocked> to tensor<1x64xf32, #blocked>
    %39 = tt.addptr %arg4, %0 : !tt.ptr<f32>, i32
    %40 = tt.splat %39 : !tt.ptr<f32> -> tensor<1x1x!tt.ptr<f32>, #blocked1>
    %41 = tt.load %40 : tensor<1x1x!tt.ptr<f32>, #blocked1>
    %42 = tt.load %40 evictionPolicy = evict_last : tensor<1x1x!tt.ptr<f32>, #blocked1>
    %43 = arith.subf %cst, %19 : tensor<1x64xf32, #blocked>
    %44 = math.exp %43 : tensor<1x64xf32, #blocked>
    %45 = arith.addf %44, %cst_3 : tensor<1x64xf32, #blocked>
    %46 = arith.divf %cst_3, %45 : tensor<1x64xf32, #blocked>
    %47 = arith.mulf %19, %46 : tensor<1x64xf32, #blocked>
    %48 = arith.mulf %15, %47 : tensor<1x64xf32, #blocked>
    %49 = arith.mulf %48, %25 : tensor<1x64xf32, #blocked>
    %50 = arith.mulf %49, %38 : tensor<1x64xf32, #blocked>
    %51 = "tt.reduce"(%50) <{axis = 1 : i32}> ({
    ^bb0(%arg9: f32, %arg10: f32):
      %85 = arith.addf %arg9, %arg10 : f32
      tt.reduce.return %85 : f32
    }) : (tensor<1x64xf32, #blocked>) -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %52 = ttg.convert_layout %51 : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<1xf32, #blocked2>
    %53 = ttg.convert_layout %52 : tensor<1xf32, #blocked2> -> tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked4}>>
    %54 = tt.expand_dims %53 {axis = 1 : i32} : tensor<1xf32, #ttg.slice<{dim = 1, parent = #blocked4}>> -> tensor<1x1xf32, #blocked4>
    %55 = ttg.convert_layout %54 : tensor<1x1xf32, #blocked4> -> tensor<1x1xf32, #blocked1>
    %56 = tt.broadcast %41 : tensor<1x1xf32, #blocked1> -> tensor<1x64xf32, #blocked1>
    %57 = ttg.convert_layout %56 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
    %58 = arith.mulf %38, %57 : tensor<1x64xf32, #blocked>
    %59 = arith.mulf %58, %25 : tensor<1x64xf32, #blocked>
    %60 = arith.mulf %15, %59 : tensor<1x64xf32, #blocked>
    %61 = arith.subf %cst_3, %46 : tensor<1x64xf32, #blocked>
    %62 = arith.mulf %19, %61 : tensor<1x64xf32, #blocked>
    %63 = arith.addf %62, %cst_3 : tensor<1x64xf32, #blocked>
    %64 = arith.mulf %46, %63 : tensor<1x64xf32, #blocked>
    %65 = arith.mulf %60, %64 : tensor<1x64xf32, #blocked>
    %66 = tt.broadcast %42 : tensor<1x1xf32, #blocked1> -> tensor<1x64xf32, #blocked1>
    %67 = ttg.convert_layout %66 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
    %68 = arith.mulf %49, %67 : tensor<1x64xf32, #blocked>
    %69 = arith.mulf %55, %cst_2 : tensor<1x1xf32, #blocked1>
    %70 = arith.mulf %42, %42 : tensor<1x1xf32, #blocked1>
    %71 = arith.mulf %70, %42 : tensor<1x1xf32, #blocked1>
    %72 = arith.mulf %69, %71 : tensor<1x1xf32, #blocked1>
    %73 = arith.mulf %72, %cst_1 : tensor<1x1xf32, #blocked1>
    %74 = arith.mulf %38, %cst_0 : tensor<1x64xf32, #blocked>
    %75 = tt.broadcast %73 : tensor<1x1xf32, #blocked1> -> tensor<1x64xf32, #blocked1>
    %76 = ttg.convert_layout %75 : tensor<1x64xf32, #blocked1> -> tensor<1x64xf32, #blocked>
    %77 = arith.mulf %76, %74 : tensor<1x64xf32, #blocked>
    %78 = arith.addf %68, %77 : tensor<1x64xf32, #blocked>
    %79 = tt.splat %arg5 : !tt.ptr<bf16> -> tensor<1x64x!tt.ptr<bf16>, #blocked>
    %80 = tt.addptr %79, %11 : tensor<1x64x!tt.ptr<bf16>, #blocked>, tensor<1x64xi32, #blocked>
    %81 = arith.truncf %65 : tensor<1x64xf32, #blocked> to tensor<1x64xbf16, #blocked>
    tt.store %80, %81 : tensor<1x64x!tt.ptr<bf16>, #blocked>
    %82 = tt.splat %arg6 : !tt.ptr<bf16> -> tensor<1x64x!tt.ptr<bf16>, #blocked>
    %83 = tt.addptr %82, %34 : tensor<1x64x!tt.ptr<bf16>, #blocked>, tensor<1x64xi32, #blocked>
    %84 = arith.truncf %78 : tensor<1x64xf32, #blocked> to tensor<1x64xbf16, #blocked>
    tt.store %83, %84 : tensor<1x64x!tt.ptr<bf16>, #blocked>
    tt.return
  }
}

{-#
  external_resources: {
    mlir_reproducer: {
      pipeline: "builtin.module(tritongpu-coalesce, tritongpu-remove-layout-conversions, tritongpu-optimize-thread-locality, tritonamdgpu-accelerate-matmul{arch-generation-name=gfx1100 kPack=1 matrix-instruction-size=0}, tritongpu-remove-layout-conversions, tritonamdgpu-optimize-epilogue, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tt.func(tritonamdgpu-hoist-layout-conversions), tritongpu-fuse-nested-loops, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, triton-licm, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritonamdgpu-stream-pipeline{global_prefetch=0 local_prefetch=0 num_stages=1 use_async_copy=false use_pingpong=false}, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, tritongpu-optimize-dot-operands{hoist-layout-conversion=true}, tritongpu-remove-layout-conversions, tritongpu-reduce-data-duplication, tritonamdgpu-reorder-instructions, tritonamdgpu-fold-true-cmpi, canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}, cse, symbol-dce)",
      disable_threading: false,
      verify_each: true
    }
  }
#-}
