// MRI command line parser
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_COMMAND_LINE_H_
#define _MRI_COMMAND_LINE_H_

#include "Resources/Phantom.h"

namespace MRI {
    namespace Science {
        class IMRISequence;
        class IMRIKernel;
    }
}

using MRI::Resources::Phantom;
using MRI::Science::IMRISequence;
using MRI::Science::IMRIKernel;

class MRICommandLine {
private:
    Phantom phantom;
    IMRISequence* sequence;
    IMRIKernel* kernel;
public:
    MRICommandLine(int argc, char* argv[]);
    virtual ~MRICommandLine();

    Phantom GetPhantom();
    IMRISequence* GetSequence();
    IMRIKernel* GetKernel();
};

#endif // _MRI_COMMAND_LINE_H_
