When I was running a piece of code, the following error appeared:

```python
Traceback (most recent call last):
  File "E:\OGM-GEpro\ACM_MM_2020\train.py", line 4, in <module>
    from dataset_3d import *
  File "E:\OGM-GEpro\ACM_MM_2020\dataset_3d.py", line 12, in <module>
    import cv2
  File "C:\...\anaconda3\Lib\site-packages\cv2\__init__.py", line 181, in <module>
    bootstrap()
  File "C:\...\anaconda3\Lib\site-packages\cv2\__init__.py", line 175, in bootstrap
    if __load_extra_py_code_for_module("cv2", submodule, DEBUG):
  File "C:\...\anaconda3\Lib\site-packages\cv2\__init__.py", line 28, in __load_extra_py_code_for_module
    py_module = importlib.import_module(module_name)
  File "C:\...\anaconda3\lib\importlib\__init__.py", line 127, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "C:\...\anaconda3\Lib\site-packages\cv2\gapi\__init__.py", line 301, in <module>
    cv.gapi.wip.GStreamerPipeline = cv.gapi_wip_gst_GStreamerPipeline
AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular import)
```

After searching online, I found the following solution:

**Comment out line 301 of the file cv2/gapi/init.py, i.e. cv.gapi.wip.GStreamerPipeline = cv.gapi_wip_gst_GStreamerPipeline. (The exact line number may differ between versions; on my computer, this line appears at the very bottom of the file.)**

This line seems to be a useless naming adjustment, and commenting it out does not affect the normal functioning of cv2. And the error is resolved by doing so.

I tested it myself, and it works.
