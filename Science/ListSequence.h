// MRI list based sequence.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#ifndef _MRI_LIST_SEQUENCE_
#define _MRI_LIST_SEQUENCE_

#include "MRISim.h"

#include <vector>

namespace MRI {
namespace Science {

using std::vector;

class ListSequence: public IMRISequence {
private:
    vector<pair<float, MRIEvent> >& seq;
    unsigned int index;
public:
    ListSequence(vector<pair<float, MRIEvent> >& seq);
    virtual ~ListSequence();
    MRIEvent GetEvent(float time);
    pair<float, MRIEvent> GetNextPoint();
    virtual void Reset(MRISim& sim);
    virtual bool HasNextPoint() const;
    void Sort();
    void Clear();
};

} // NS Science
} // NS MRI

#endif // _MRI_LIST_SEQUENCE_
