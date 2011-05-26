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


#include <Utils/PropertyTree.h>
#include <Utils/PropertyTreeNode.h>

namespace MRI {
namespace Science {

using namespace OpenEngine::Utils;

ListSequence::ListSequence()
    : index(0)
{
        
}
    
ListSequence::~ListSequence() {

}

// warning: for now do not mix calls to GetEvent with calls to GetNextPoint;
// MRIEvent ListSequence::GetEvent(float time) {
//     MRIEvent state;
//     if (index >= seq.size()) return state;
//     if (seq[index].first <= time)
//         return seq[index++].second;
//     return state;
// }

// warning: for now do not mix calls to GetEvent with calls to GetNextPoint;
pair<double, MRIEvent> ListSequence::GetNextPoint() {
    if (index == seq.size()) throw Exception("no more time points");
    return seq[index++];
}

bool ListSequence::HasNextPoint() const {
    return !(index == seq.size());
}

// vector<pair<double, MRIEvent> > ListSequence::GetPoints() {
//     return seq;
// }

unsigned int ListSequence::GetNumPoints() {
    return seq.size();
}

double ListSequence::GetDuration() {
    if (seq.empty()) 
        return 0.0;
    return seq[seq.size()-1].first - seq[0].first;
}

void ListSequence::Reset(MRISim& sim) {
    index = 0;
}

bool sortf (pair<float, MRIEvent> i, pair<float, MRIEvent> j) { return (i.first < j.first); }

void ListSequence::Sort() {
    sort(seq.begin(), seq.end(), sortf);
}

void ListSequence::Clear() {
    seq.clear();
    index = 0;
}


void ListSequence::LoadFromYamlFile(string file) {
    logger.info << "start loading file: " << file << logger.end;
    PropertyTree tree(file);
    Clear();

    PropertyTreeNode* steps = tree.GetRootNode();
    unsigned int size = steps->GetSize();
    logger.info << "steps: " << size << logger.end;
    for (unsigned int i = 0; i < size; ++i) {
        PropertyTreeNode* step = steps->GetNodeIdx(i);        

        if (!step->HaveNodePath("time"))
            throw Exception("no time in step");
        double time = step->GetPath("time", double(0.0));

        if (!step->HaveNode("actions"))
            throw Exception("no actions in step");
        
        MRIEvent e;
        PropertyTreeNode* actions = step->GetNodePath("actions");
        // logger.info << "actions: " << actions->GetSize() << logger.end;
        for (unsigned int j = 0; j < actions->GetSize(); ++j) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            string name = action->GetPath("name", string());
            // logger.info << "read name: " << name << logger.end;
            if (name == string("RECORD")) {

                e.action |= MRIEvent::RECORD;
                e.point[0] = action->GetPath("point-x", (unsigned int)(0));
                e.point[1] = action->GetPath("point-y", (unsigned int)(0));
                e.point[2] = action->GetPath("point-z", (unsigned int)(0));
            }

            if (name == string("RESET")) {
                e.action |= MRIEvent::RESET;
            }

            if (name == string("EXCITE")) {
                e.action |= MRIEvent::EXCITE;
                e.angleRF = action->GetPath("angleRF", float(0.0));
                e.slice = action->GetPath("slice", (unsigned int)(0));
            }

            if (name == string("GRADIENT")) {
                e.action |= MRIEvent::GRADIENT;
                e.gradient[0] = action->GetPath("gradient-x", float(0.0));
                e.gradient[1] = action->GetPath("gradient-y", float(0.0));
                e.gradient[2] = action->GetPath("gradient-z", float(0.0));
            }

            if (name == string("RFPULSE")) {
                logger.info << "in pulse" << logger.end;
                e.action |= MRIEvent::RFPULSE;
                e.rfSignal[0] = action->GetPath("rf-x", float(0.0));
                e.rfSignal[1] = action->GetPath("rf-y", float(0.0));
                e.rfSignal[2] = action->GetPath("rf-z", float(0.0));
                e.dt = step->GetPath("dt", float(0.0));
            }

            if (name == string("DONE")) {
                e.action |= MRIEvent::DONE;
            }
        }

        seq.push_back(make_pair(time, e));
    }
    logger.info << "done loading file" << logger.end;
}

void ListSequence::LoadFromYamlFile() {
    LoadFromYamlFile("sequence.yaml");
}

void ListSequence::SaveToYamlFile(string file) {
    PropertyTree tree;
    PropertyTreeNode* steps = tree.GetRootNode();

    vector<pair<double, MRIEvent> >::iterator itr = seq.begin();
    unsigned int i = 0;
    for (; itr != seq.end(); ++itr) {
        double time = (*itr).first;
        MRIEvent e = (*itr).second;
        
        //step node
        PropertyTreeNode* step = steps->GetNodeIdx(i);
        step->SetPath("time", time);

        PropertyTreeNode* actions = step->GetNodePath("actions");

        unsigned int j = 0;
        if (e.action & MRIEvent::RECORD) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            action->SetPath("name", "RECORD");
            action->SetPath("point-x", e.point[0]);
            action->SetPath("point-y", e.point[1]);
            action->SetPath("point-z", e.point[2]);
            ++j;
        }

        if (e.action & MRIEvent::RESET) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            action->SetPath("name", "RESET");
            ++j;
        }
        
        if (e.action & MRIEvent::EXCITE) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            action->SetPath("name", "EXCITE");
            action->SetPath("angleRF", e.angleRF);
            action->SetPath("slice", e.slice);
            ++j;
        }

        if (e.action & MRIEvent::GRADIENT) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            action->SetPath("name", "GRADIENT");
            action->SetPath("gradient-x", e.gradient[0]);
            action->SetPath("gradient-y", e.gradient[1]);
            action->SetPath("gradient-z", e.gradient[2]);
            ++j;
        }

        if (e.action & MRIEvent::RFPULSE) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            action->SetPath("name", "RFPULSE");
            action->SetPath("rf-x", e.rfSignal[0]);
            action->SetPath("rf-y", e.rfSignal[1]);
            action->SetPath("rf-z", e.rfSignal[2]);
            step->SetPath("dt", e.dt);
            ++j;
        }

        if (e.action & MRIEvent::DONE) {
            PropertyTreeNode* action = actions->GetNodeIdx(j);
            action->SetPath("name", "DONE");
            ++j;
        }
        ++i;
    }
    tree.SaveToFile(file);
}

void ListSequence::SaveToYamlFile() {
    SaveToYamlFile("sequence.yaml");
}


ValueList ListSequence::Inspect() {
    ValueList values;
    
    {
        ActionValueCall<ListSequence> *v =
            new ActionValueCall<ListSequence>(*this, &ListSequence::SaveToYamlFile);
            v->name = "Save to file";
            values.push_back(v);
    }

    {
        ActionValueCall<ListSequence> *v =
            new ActionValueCall<ListSequence>(*this, &ListSequence::LoadFromYamlFile);
            v->name = "Load from file";
            values.push_back(v);
    }
    return values;
}

} // NS Science
} // NS MRI
