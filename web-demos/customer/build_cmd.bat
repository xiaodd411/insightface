
-- 使用打包为单个文件的话再程序启动时会有解压文件到临时目录的操作,会导致程序运行耗时教程
pyinstaller --onefile --collect-binaries=onnx --hidden-import=onnx --hidden-import=onnx_cpp2py_export --add-binary "C:\Users\xq\Desktop\project\insightface\venv1\Lib\site-packages\onnx\onnx_cpp2py_export.cp311-win_amd64.pyd;onnx" main.py


--  使用目录打包的话会减少程序执行时所耗时间,但是打包后的文件会变大
pyinstaller --onedir --collect-binaries=onnx --hidden-import=onnx --hidden-import=onnx_cpp2py_export --add-binary "C:\Users\xq\Desktop\project\insightface\venv1\Lib\site-packages\onnx\onnx_cpp2py_export.cp311-win_amd64.pyd;onnx" main.py

pyinstaller --onedir --name=insightface --collect-binaries=onnx --hidden-import=onnx --hidden-import=onnx_cpp2py_export --add-binary "C:\Users\xq\Desktop\project\insightface\venv1\Lib\site-packages\onnx\onnx_cpp2py_export.cp311-win_amd64.pyd;onnx" .\main.py
