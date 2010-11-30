// MRI list based sequence.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "ListSequence.h"

namespace MRI {
namespace Science {

ListSequence::ListSequence(vector<pair<float, MRIEvent> >& seq)
    : seq(seq), index(0)
{
        
}
    
ListSequence::~ListSequence() {

}

MRIEvent ListSequence::GetEvent(float time) {
    MRIEvent state;
    if (index >= seq.size()) return state;
    if (seq[index].first <= time)
        return seq[index++].second;
    return state;
}

} // NS Science
} // NS MRI
