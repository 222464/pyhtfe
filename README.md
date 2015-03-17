# ![pyHTFE Logo](http://i1218.photobucket.com/albums/dd401/222464/PYHTFELOGOSMALL.png)

pyHTFE
=======

A derivative of HTM (hierarchical temporal memory) geared towards high-performance GPU sequence prediction.

Install
-----------

pyHTFE relies on OpenCL.

To get OpenCL, refer to your graphics hardware vendor website (for AMD and Nvidia), or CPU vendor (e.g. the Intel OpenCL SDK).
Works best with AMD cards (best OpenCL support).

Once OpenCL is installed, navigate to the setup.py file contained in the source directory. Run the following: 

```python
python setup.py install
```

If this doesn't immediately work, it is possible that the OpenCL SDK installed to an unexpected location. If this is the case, open setup.py and change the following lines to match the location of the OpenCL SDK, and try installing again:

```python
clIncludeDir = "C:/Program Files (x86)/AMD APP SDK/3.0-0-Beta/include/"
clLibDir = "C:/Program Files (x86)/AMD APP SDK/3.0-0-Beta/lib/x86_64/"
```

Usage
-----------

More detailed usage instructions will come. For now, see the example.py for usage.

License
-----------

pyHTFE
Copyright (C) 2015 Eric Laukien

This software is provided 'as-is', without any express or implied
warranty.  In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
	claim that you wrote the original software. If you use this software
	in a product, an acknowledgment in the product documentation would be
	appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
	misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.

------------------------------------------------------------------------------

pyHTFE uses the following external libraries:

OpenCL

