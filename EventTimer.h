//
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS)
//
// This program is free software; It is covered by the GNU General
// Public License version 2 or any later version.
// See the GNU General Public License for more details (see LICENSE).
//--------------------------------------------------------------------


#ifndef _OE_EVENT_TIMER_H_
#define _OE_EVENT_TIMER_H_

#include <Logging/Logger.h>
#include <Core/EngineEvents.h>

#include <Core/IListener.h>
#include <Core/Event.h>

#include <Utils/Timer.h>

namespace MRI {

using namespace OpenEngine::Core;
using namespace OpenEngine::Utils;

using OpenEngine::Core::ProcessEventArg;

class EventTimer;

class TimerEventArg {
public:
    EventTimer* et;
    TimerEventArg(EventTimer* et) : et(et) {}
};

class EventTimer : IListener<ProcessEventArg> {
    Event<TimerEventArg> timerEvent;
    Timer t;
    float time;
public:
    EventTimer(float time) : time(time) {
        t.Start();
    }

    void Handle(ProcessEventArg arg) {
        if (t.GetElapsedTime().AsInt() / 1000000.0 > time ) {
            timerEvent.Notify(TimerEventArg(this));
            t.Reset();
        }
    }

    IEvent<TimerEventArg>& TimerEvent() {
        return timerEvent;
    }
};

} // NS MRI

#endif // _OE_EVENT_TIMER_H_
