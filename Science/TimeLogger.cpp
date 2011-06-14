// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------

#include "TimeLogger.h"

using namespace std;
using namespace OpenEngine::Utils;

namespace MRI {
namespace Science {

    TimeLogger::TimeLogger(string fname) {
        fout = new fstream(fname.c_str(), fstream::out 
                           | fstream::trunc);
        total = 0.0;
        
                           
    }

    TimeLogger::~TimeLogger() {
        fout->close();
        delete fout;
    }

    void TimeLogger::Start() {
        timer.Reset();
        timer.Start();
    }
    void TimeLogger::Stop() {
        timer.Stop();
        total += double(timer.GetElapsedIntervals(1)) * 1e-6;
        *fout  << timer.GetElapsedIntervals(1) << std::endl;
    }

    double TimeLogger::GetTotal() {
        return total;
    }



} // NS Science
} // NS OpenEngine

