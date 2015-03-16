%begin %{
#include <cmath>
#include <iostream>
%}
%module htfe

%{
#include "htfe/HTFE.h"
%}

%include "std_string.i"
%include "std_vector.i"

namespace std {
   %template(vectorld) vector<htfe::LayerDesc>;
};

%include "htfe/HTFE.h"
%include "system/ComputeSystem.h"
%include "system/ComputeProgram.h"