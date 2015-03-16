from distutils.core import setup, Extension

# Change these to point to the right locations
clIncludeDir = "C:/Program Files (x86)/AMD APP SDK/3.0-0-Beta/include/"
clLibDir = "C:/Program Files (x86)/AMD APP SDK/3.0-0-Beta/lib/x86_64/"

extension_mod = Extension(name="_htfe", sources=["HTFE.i", "system/ComputeSystem.cpp", "system/ComputeProgram.cpp", "htfe/HTFE.cpp"], swig_opts=["-c++"], language=["c++"], include_dirs=[clIncludeDir, "./"], library_dirs=[clLibDir], libraries=["OpenCL"])

setup(name = "htfe", version="1.0", ext_modules=[extension_mod], package_data={"htfe": ["../resources/*.cl"]})