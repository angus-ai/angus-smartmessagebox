# -*- coding: utf-8 -*-

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import StringIO
import base64
import datetime
import os
import random
import subprocess
import time
import uuid
import wave
import zlib
import sys

from toolbox import AudioService
import angus
import cv2
import numpy as np
import pyaudio


class What(AudioService):
    """
    Record sound and stops at keyword detection.
    This class use angus.ai to spot the keyword among chunks of sound
    """

    def __init__(self, device, angus_root, vocabs, service_version=1):
        super(What, self).__init__(device)

        # create a proxy on angus service with the given version
        self.service = angus_root.services.get_service(
            "word_spotting",
            service_version)

        self.vocabs = vocabs

    def call(self, timeout=3, callback=None):
        """
        Record sound until a keyword is spotted or timeout is reached, call
        the callback with the audio wav, the original wav temporarly file path and
        the resampled temorarly file path
        """

        # you must enable session feature with the vocabulary
        self.service.enable_session({"vocabulary": self.vocabs,
                                     "lang": "en-US",
                                     })

        # the callback get the audio buffer and send it to angus to check
        # keyword
        def callback_(buff):
            tmpfile = str(uuid.uuid1())

            input_path = "/tmp/%s.wav" % (tmpfile)
            output_path = "/tmp/%s.out.wav" % (tmpfile)

            with open(input_path, "wb") as f:
                f.write(buff)

            # angus.ai work with audio sample rate of 16000, we must
            # resample the sound
            subprocess.call(["sox", input_path, "-r", "16000", output_path])

            if callback:
                callback(buff, input_path, output_path)

            # call angus.ai
            job = self.service.process(
                {'sound': open(output_path, "rb"), 'sensitivity': 0.9})

            # check if a keyword was spotted
            if "nbests" in job.result:
                print job.result["nbests"]
                if job.result["nbests"][0]["confidence"] > 0.15:
                    result = job.result["nbests"][0]["words"]
                    return result

        # run record with the angus.ai callback
        result = self.record(timeout, callback_)

        # close the session
        self.service.disable_session()

        return result


class RecordUntil(What):
    """
    Record the audio stream in a file until a keyword is spotted or timeout
    is reached
    """

    def __init__(self, *args, **kwargs):
        super(RecordUntil, self).__init__(*args, **kwargs)

    def call(self, timeout):
        # keep back all temporarly original audio files
        chunks = []

        def callback(buff, input_path, output_path):
            chunks.append(input_path)

        super(RecordUntil, self).call(timeout, callback)

        # concatenate all temporarly original audio files into a single one
        # define a new output file
        path = "/tmp/%s.wav" % (str(uuid.uuid1()))
        output = wave.open(path, "w")

        if len(chunks) == 0:
            return None

        first = True
        # for each files
        for i in chunks:
            w = wave.open(i, "rb")
            if first:
                output.setparams(w.getparams())
                first = False
            # concatenate the content
            output.writeframes(w.readframes(w.getnframes()))
        # close the output file
        output.close()

        # and return the path of the output file
        return path


class Play(AudioService):
    """
    Play a wav.
    """

    def __init__(self, device):
        super(Play, self).__init__(device)

    def call(self, path):
        self.play(wave.open(path, 'r'))


class Say(AudioService):
    """
    Call the angus.ai tts and play the result.
    """

    def __init__(self, device, angus_root, lang="en-US"):
        super(Say, self).__init__(device)
        self.service = angus_root.services.get_service("text_to_speech")
        self.lang = lang

    def call(self, text):
        if not text:
            return
        job = self.service.process({
            "text": text,
            "lang": self.lang,
        })

        # the result was encoded in base64 and compress with zlib
        data = job.result["sound"]
        data = base64.b64decode(data)
        data = zlib.decompress(data)

        # create an in-memory file to read the wav file
        f = wave.open(StringIO.StringIO(data), 'r')

        # play the data
        self.play(f)


class Whois(object):
    """
    Return the reconized face or None if timeout
    """

    def __init__(self, device, angus_root, album):
        self.angus_root = angus_root
        self.device = device

        # we must create an album to identify user
        angus_album = dict()
        for name in album:
            resources = [
                angus_root.blobs.create(
                    open(
                        face,
                        "rb")) for face in album[name]]
            angus_album[name] = resources
        self.album = angus_album
        self.service = self.angus_root.services.get_service("face_recognition")

    def call(self, timeout=5):
        # create a session with the identification album
        self.service.enable_session({"album": self.album})

        # use opencv to create a video stream
        cap = cv2.VideoCapture(self.device)

        # set width and height, 640x480 is the minimum
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, 480)

        # get the start timestamp
        start = time.time()
        result = None

        while cap.isOpened() and (time.time() - start) < timeout:
            # read a frame
            ret, frame = cap.read()
            if frame is not None:

                # convert frame into back and white image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # compress it into png
                ret, buff = cv2.imencode(".png", gray)

                # extract binary data and create an in-memory file to call
                # angus.ai face recognition service
                buff = np.array(buff).tostring()
                job = self.service.process({
                    "image": StringIO.StringIO(buff)
                })

                # check the result, if a user was reconized, break and return
                # the result
                if job.result["nb_faces"] > 0:
                    name = job.result["faces"][0]["names"][0]
                    if name["confidence"] > 0.1:
                        result = name["key"]
                        break
        # close the session
        self.service.disable_session()

        # return the result, None if no one.
        return result


def retry(n, f, *args, **kwargs):
    for i in range(n):
        r = f(*args, **kwargs)
        if r:
            return r
        else:
            say.call(u"Sorry, could you repeat please ?")
    return None


class Message(object):

    def __init__(self, who, forwho, wav, timestamp=None):
        if not timestamp:
            self.timestamp = datetime.datetime.now()
        else:
            self.timestamp = timestamp

        self.who = who
        self.forwho = forwho
        self.wav = wav

    def __del__(self):
        os.remove(self.wav)


MSGS = dict()


def record_msgs(what, say, whois, who, play, user):
    say.call("Who is the recipient ?")
    forwho = retry(4, who.call, timeout=10)
    if not forwho:
        return

    say.call("You can start to leave your message and finalize it by stop.")
    wav_path = recorder.call(90)
    say.call("Now, I play the recorded message:")
    play.call(wav_path)
    say.call("Is the message correct ?")
    if retry(3, what.call, timeout=5) == "yes":
        say.call("The new message is now saved to %s." % forwho)
        MSGS.setdefault(forwho, []).append(Message(user, forwho, wav_path))

    say.call("Alright, thank you and see you soon.")


def read_msgs(what, say, whois, who, play, msgs):
    while msgs:
        msg = msgs.pop()

        while True:
            say.call(
                "This is a message record a %s at %s" %
                (msg.who, msg.timestamp.strftime("%H:%M")))
            play.call(msg.wav)
            say.call("Do you want play it again ?")
            if retry(3, what.call, timeout=5) == "no":
                break

    say.call("End of the new message, thank you.")


def main_behavior(what, say, whois, who, play):
    INTROS = [
        "Excuse-me ?",
        "Com'on",
        "Hello !",
        "I don't see you very well",
    ]

    say.call(random.choice(INTROS))
    user = whois.call(timeout=30)
    if user:
        say.call("Hello %s !" % user)

        msgs = MSGS.get(user, [])

        if msgs:
            say.call(
                "You have %s new messages, do you want listen them ?" %
                (len(msgs)))
            if retry(3, what.call, timeout=5) == "yes":
                read_msgs(what, say, whois, who, play, msgs)
        else:
            say.call("You have no new message.")

        say.call("Do you want record a new message ?")

        if retry(3, what.call, timeout=5) == "yes":
            record_msgs(what, say, whois, who, play, user)
        else:
            say.call("Alright, have a good day, see you soon.")

def choose_io(inputs=False, outputs=False):
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if ((inputs and dev["maxInputChannels"] > 0) or
            (outputs and dev["maxOutputChannels"] >0)):
            print("%s : %s"%(i, dev["name"]))

    if inputs and outputs:
        msg = "i/o"
    elif inputs:
        msg = "input"
    elif outputs:
        msg = "output"

    d = int(raw_input("Select your %s: "%msg))

    return d

if __name__ == "__main__":
    ########
    # Angus
    ########
    root = angus.connect()

    ################
    # Input / Output
    ################
    if len(sys.argv) == 3:
        audio_in = int(sys.argv[1])
        audio_out = int(sys.argv[2])
    else:
        audio_in = choose_io(inputs=True)
        audio_out = choose_io(outputs=True)

    ###########
    # Directory
    ###########
    directory = dict()
    for name in os.listdir("ids"):
        for face in os.listdir("ids/%s"%(name)):
            if face.endswith(".jpg") or face.endswidth(".png"):
                directory.setdefault(name, []).append("ids/%s/%s"%(name, face))

    # define vocabulary to spot the names of targets
    targets = [{"words": key} for key in directory]

    # define vocabulary for interaction
    yesno = [ { "words": "yes"}, { "words": "no" } ]
    stop  = [ { "words": "stop" } ]

    ##########
    # Services
    ##########

    whois = Whois(0, root, directory)
    say = Say(audio_out, root)
    what = What(audio_in, root, yesno, service_version=2)
    recorder = RecordUntil(audio_in, root, stop, service_version=2)
    play = Play(audio_out)
    who = What(audio_in, root, targets, service_version=2)

    ###########
    # Main loop
    ###########

    say.call(
        "Hello, please put you well on video camera so I can recognize you")

    while True:
        main_behavior(what, say, whois, who, play)
