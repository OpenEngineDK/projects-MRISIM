// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_TIME_LOGGER_H_
#define _OE_TIME_LOGGER_H_

#include <fstream>
#include <string>
#include <Utils/Timer.h>

namespace MRI {
namespace Science {

/**
 * Short description.
 *
 * @class TimeLogger TimeLogger.h s/MRISIM/Science/TimeLogger.h
 */
class TimeLogger {
private:
    std::fstream* fout;
    OpenEngine::Utils::Timer timer;
public:
    TimeLogger(std::string fname);
    ~TimeLogger();
    void Start();
    void Stop();
};


} // NS Science
} // NS OpenEngine

#endif // _OE_TIME_LOGGER_H_
