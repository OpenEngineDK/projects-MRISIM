// MRI list based sequence.
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "ListSequence.h"

#include <algorithm>

namespace MRI {
namespace Science {

ListSequence::ListSequence(vector<pair<float, MRIEvent> >& seq)
    : seq(seq), index(0)
{
        
}
    
ListSequence::~ListSequence() {

}

// warning: for now do not mix calls to GetEvent with calls to GetNextPoint;
MRIEvent ListSequence::GetEvent(float time) {
    MRIEvent state;
    if (index >= seq.size()) return state;
    if (seq[index].first <= time)
        return seq[index++].second;
    return state;
}

// warning: for now do not mix calls to GetEvent with calls to GetNextPoint;
pair<float, MRIEvent> ListSequence::GetNextPoint() {
    if (index == seq.size()) throw Exception("no more time points");
    return seq[index++];
}

void ListSequence::Reset() {
    index = 0;
}

bool sortf (pair<float, MRIEvent> i, pair<float, MRIEvent> j) { return (i.first < j.first); }

void ListSequence::Sort() {
    sort(seq.begin(), seq.end(), sortf);
}

} // NS Science
} // NS MRI
