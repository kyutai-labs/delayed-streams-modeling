# FAQ

Here is the answer to a number of frequently asked questions.

### Torch Compilation Errors

With some PyTorch/triton versions, one might encounter compilation errors
like the following:
```
  Traceback (most recent call last):
  ...
  File "site-packages/torch/_inductor/runtime/triton_heuristics.py", line 1153, in make_launcher
    "launch_enter_hook": binary.__class__.launch_enter_hook,
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch._inductor.exc.InductorError: AttributeError: type object 'CompiledKernel' has no attribute 'launch_enter_hook'
```

If that's the case, you can disable torch compilation by setting the following
environment variable.
```bash
export NO_TORCH_COMPILE=1
```

### Will you release training code?

Some finetuning code can be found in the [kyutai-labs/moshi-finetune repo](https://github.com/kyutai-labs/moshi-finetune).
This code has not been adapted to the Speech-To-Text and Text-To-Speech models
yet, but it should be a good starting point.


