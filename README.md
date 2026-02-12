# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/scalar.py minitorch/scalar_functions.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/tensor.py minitorch/datasets.py minitorch/testing.py minitorch/optim.py

```python
python project/parallel_check.py
```

```bash
MAP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_map.<locals>._map, 
/workspace/workspace/Module-3/minitorch/fast_ops.py (162)  
================================================================================


Parallel loop listing for  Function tensor_map.<locals>._map, /workspace/workspace/Module-3/minitorch/fast_ops.py (162) 
------------------------------------------------------------------------------------------|loop #ID
    def _map(                                                                             | 
        out: Storage,                                                                     | 
        out_shape: Shape,                                                                 | 
        out_strides: Strides,                                                             | 
        in_storage: Storage,                                                              | 
        in_shape: Shape,                                                                  | 
        in_strides: Strides,                                                              | 
    ) -> None:                                                                            | 
        # TODO: Implement for Task 3.1.                                                   | 
        size = int(np.prod(out_shape))----------------------------------------------------| #2
        for i in prange(size):------------------------------------------------------------| #3
            out_index = np.zeros(len(out_shape), dtype=np.int32)--------------------------| #0
            in_index = np.zeros(len(in_shape), dtype=np.int32)----------------------------| #1
            same = len(out_shape) == len(in_shape)                                        | 
            if same:                                                                      | 
                for k in range(len(out_shape)):                                           | 
                    if out_shape[k] != in_shape[k] or out_strides[k] != in_strides[k]:    | 
                        same = False                                                      | 
                        break                                                             | 
            if same:                                                                      | 
                out[i] = fn(in_storage[i])                                                | 
                continue                                                                  | 
            to_index(i, out_shape, out_index)                                             | 
            broadcast_index(out_index, out_shape, in_shape, in_index)                     | 
            in_pos = index_to_position(in_index, in_strides)                              | 
            out_pos = index_to_position(out_index, out_strides)                           | 
            out[out_pos] = fn(in_storage[in_pos])                                         | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 4 parallel for-
loop(s) (originating from loops labelled: #2, #3, #0, #1).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--3 is a parallel loop
   +--0 --> rewritten as a serial loop
   +--1 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (parallel)
   +--1 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--3 (parallel)
   +--0 (serial)
   +--1 (serial)


 
Parallel region 0 (loop #3) had 0 loop(s) fused and 2 loop(s) serialized as part
 of the larger parallel loop (#3).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (173) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (174) is hoisted out of the 
parallel loop labelled #3 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: in_index = np.zeros(len(in_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
ZIP
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_zip.<locals>._zip, 
/workspace/workspace/Module-3/minitorch/fast_ops.py (216)  
================================================================================


Parallel loop listing for  Function tensor_zip.<locals>._zip, /workspace/workspace/Module-3/minitorch/fast_ops.py (216) 
-----------------------------------------------------------------------|loop #ID
    def _zip(                                                          | 
        out: Storage,                                                  | 
        out_shape: Shape,                                              | 
        out_strides: Strides,                                          | 
        a_storage: Storage,                                            | 
        a_shape: Shape,                                                | 
        a_strides: Strides,                                            | 
        b_storage: Storage,                                            | 
        b_shape: Shape,                                                | 
        b_strides: Strides,                                            | 
    ) -> None:                                                         | 
        # TODO: Implement for Task 3.1.                                | 
        size = int(np.prod(out_shape))---------------------------------| #7
        for i in prange(size):-----------------------------------------| #8
            out_index = np.zeros(len(out_shape), dtype=np.int32)-------| #4
            a_index = np.zeros(len(a_shape), dtype=np.int32)-----------| #5
            b_index = np.zeros(len(b_shape), dtype=np.int32)-----------| #6
            same = (                                                   | 
                len(out_shape) == len(a_shape)                         | 
                and len(out_shape) == len(b_shape)                     | 
            )                                                          | 
            if same:                                                   | 
                for k in range(len(out_shape)):                        | 
                    if (                                               | 
                        out_shape[k] != a_shape[k]                     | 
                        or out_shape[k] != b_shape[k]                  | 
                        or out_strides[k] != a_strides[k]              | 
                        or out_strides[k] != b_strides[k]              | 
                    ):                                                 | 
                        same = False                                   | 
                        break                                          | 
            if same:                                                   | 
                out[i] = fn(a_storage[i], b_storage[i])                | 
                continue                                               | 
            to_index(i, out_shape, out_index)                          | 
            broadcast_index(out_index, out_shape, a_shape, a_index)    | 
            broadcast_index(out_index, out_shape, b_shape, b_index)    | 
            a_pos = index_to_position(a_index, a_strides)              | 
            b_pos = index_to_position(b_index, b_strides)              | 
            out_pos = index_to_position(out_index, out_strides)        | 
            out[out_pos] = fn(a_storage[a_pos], b_storage[b_pos])      | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 5 parallel for-
loop(s) (originating from loops labelled: #7, #8, #4, #5, #6).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--8 is a parallel loop
   +--4 --> rewritten as a serial loop
   +--5 --> rewritten as a serial loop
   +--6 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (parallel)
   +--5 (parallel)
   +--6 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--8 (parallel)
   +--4 (serial)
   +--5 (serial)
   +--6 (serial)


 
Parallel region 0 (loop #8) had 0 loop(s) fused and 3 loop(s) serialized as part
 of the larger parallel loop (#8).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (230) is hoisted out of the 
parallel loop labelled #8 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (231) is hoisted out of the 
parallel loop labelled #8 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (232) is hoisted out of the 
parallel loop labelled #8 (it will be performed before the loop is executed and 
reused inside the loop):
   Allocation:: b_index = np.zeros(len(b_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
REDUCE
 
================================================================================
 Parallel Accelerator Optimizing:  Function tensor_reduce.<locals>._reduce, 
/workspace/workspace/Module-3/minitorch/fast_ops.py (282)  
================================================================================


Parallel loop listing for  Function tensor_reduce.<locals>._reduce, /workspace/workspace/Module-3/minitorch/fast_ops.py (282) 
---------------------------------------------------------------------|loop #ID
    def _reduce(                                                     | 
        out: Storage,                                                | 
        out_shape: Shape,                                            | 
        out_strides: Strides,                                        | 
        a_storage: Storage,                                          | 
        a_shape: Shape,                                              | 
        a_strides: Strides,                                          | 
        reduce_dim: int,                                             | 
    ) -> None:                                                       | 
        # TODO: Implement for Task 3.1.                              | 
        size = int(np.prod(out_shape))-------------------------------| #11
        for i in prange(size):---------------------------------------| #12
            out_index = np.zeros(len(out_shape), dtype=np.int32)-----| #9
            to_index(i, out_shape, out_index)                        | 
            out_pos = index_to_position(out_index, out_strides)      | 
            for j in range(a_shape[reduce_dim]):                     | 
                a_index = np.zeros(len(a_shape), dtype=np.int32)-----| #10
                for k in range(len(out_shape)):                      | 
                    if k == reduce_dim:                              | 
                        a_index[k] = j                               | 
                    else:                                            | 
                        a_index[k] = out_index[k]                    | 
                a_pos = index_to_position(a_index, a_strides)        | 
                out[out_pos] = fn(out[out_pos], a_storage[a_pos])    | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 3 parallel for-
loop(s) (originating from loops labelled: #11, #12, #9).
--------------------------------------------------------------------------------
---------------------------- Optimising loop nests -----------------------------
Attempting loop nest rewrites (optimising for the largest parallel loops)...
 
+--12 is a parallel loop
   +--9 --> rewritten as a serial loop
   +--10 --> rewritten as a serial loop
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (parallel)
   +--10 (parallel)


--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel region 0:
+--12 (parallel)
   +--9 (serial)
   +--10 (serial)


 
Parallel region 0 (loop #12) had 0 loop(s) fused and 2 loop(s) serialized as 
part of the larger parallel loop (#12).
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (298) is hoisted out of the 
parallel loop labelled #12 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: a_index = np.zeros(len(a_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
The memory allocation derived from the instruction at 
/workspace/workspace/Module-3/minitorch/fast_ops.py (294) is hoisted out of the 
parallel loop labelled #12 (it will be performed before the loop is executed and
 reused inside the loop):
   Allocation:: out_index = np.zeros(len(out_shape), dtype=np.int32)
    - numpy.empty() is used for the allocation.
None
MATRIX MULTIPLY
 
================================================================================
 Parallel Accelerator Optimizing:  Function _tensor_matrix_multiply, 
/workspace/workspace/Module-3/minitorch/fast_ops.py (310)  
================================================================================


Parallel loop listing for  Function _tensor_matrix_multiply, /workspace/workspace/Module-3/minitorch/fast_ops.py (310) 
--------------------------------------------------------------------------------------------|loop #ID
def _tensor_matrix_multiply(                                                                | 
    out: Storage,                                                                           | 
    out_shape: Shape,                                                                       | 
    out_strides: Strides,                                                                   | 
    a_storage: Storage,                                                                     | 
    a_shape: Shape,                                                                         | 
    a_strides: Strides,                                                                     | 
    b_storage: Storage,                                                                     | 
    b_shape: Shape,                                                                         | 
    b_strides: Strides,                                                                     | 
) -> None:                                                                                  | 
    """NUMBA tensor matrix multiply function.                                               | 
                                                                                            | 
    Should work for any tensor shapes that broadcast as long as                             | 
                                                                                            | 
    ```                                                                                     | 
    assert a_shape[-1] == b_shape[-2]                                                       | 
    ```                                                                                     | 
                                                                                            | 
    Optimizations:                                                                          | 
                                                                                            | 
    * Outer loop in parallel                                                                | 
    * No index buffers or function calls                                                    | 
    * Inner loop should have no global writes, 1 multiply.                                  | 
                                                                                            | 
                                                                                            | 
    Args:                                                                                   | 
    ----                                                                                    | 
        out (Storage): storage for `out` tensor                                             | 
        out_shape (Shape): shape for `out` tensor                                           | 
        out_strides (Strides): strides for `out` tensor                                     | 
        a_storage (Storage): storage for `a` tensor                                         | 
        a_shape (Shape): shape for `a` tensor                                               | 
        a_strides (Strides): strides for `a` tensor                                         | 
        b_storage (Storage): storage for `b` tensor                                         | 
        b_shape (Shape): shape for `b` tensor                                               | 
        b_strides (Strides): strides for `b` tensor                                         | 
                                                                                            | 
    Returns:                                                                                | 
    -------                                                                                 | 
        None : Fills in `out`                                                               | 
                                                                                            | 
    """                                                                                     | 
    a_batch_stride = a_strides[0] if a_shape[0] > 1 else 0                                  | 
    b_batch_stride = b_strides[0] if b_shape[0] > 1 else 0                                  | 
                                                                                            | 
    # TODO: Implement for Task 3.2.                                                         | 
    batch = out_shape[0]                                                                    | 
    M = out_shape[-2]                                                                       | 
    N = out_shape[-1]                                                                       | 
    K = a_shape[-1]                                                                         | 
    for n in prange(batch):-----------------------------------------------------------------| #13
        for i in range(M):                                                                  | 
            for j in range(N):                                                              | 
                acc = 0.0                                                                   | 
                for k in range(K):                                                          | 
                    a_pos = n * a_batch_stride + i * a_strides[-2] + k * a_strides[-1]      | 
                    b_pos = n * b_batch_stride + k * b_strides[-2] + j * b_strides[-1]      | 
                    acc += a_storage[a_pos] * b_storage[b_pos]                              | 
                out_pos = n * out_strides[0] + i * out_strides[-2] + j * out_strides[-1]    | 
                out[out_pos] = acc                                                          | 
--------------------------------- Fusing loops ---------------------------------
Attempting fusion of parallel loops (combines loops with similar properties)...
Following the attempted fusion of parallel for-loops there are 1 parallel for-
loop(s) (originating from loops labelled: #13).
--------------------------------------------------------------------------------
----------------------------- Before Optimisation ------------------------------
--------------------------------------------------------------------------------
------------------------------ After Optimisation ------------------------------
Parallel structure is already optimal.
--------------------------------------------------------------------------------
--------------------------------------------------------------------------------
 
---------------------------Loop invariant code motion---------------------------
Allocation hoisting:
No allocation hoisting found
None
```