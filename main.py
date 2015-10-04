# -*- coding: utf-8 -*-
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

        # create a vocabulary dictionnary for word spotting service
        angus_vocabs = dict()
        for vocab in vocabs:
            resources = [
                angus_root.blobs.create(
                    open(
                        v,
                        "rb")) for v in vocabs[vocab]]
            angus_vocabs[vocab] = resources
        self.vocabs = angus_vocabs

    def call(self, timeout=3, callback=None):
        """
        Record sound until a keyword is spotted or timeout is reached, call
        the callback with the audio wav, the original wav temporarly file path and
        the resampled temorarly file path
        """

        # you must enable session feature with the vocabulary
        self.service.enable_session({"vocabulary": self.vocabs,
                                     "lang": "fr-FR",
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
                {'sound': open(output_path, "rb"), 'sensitivity': 0.7})

            # check if a keyword was spotted
            if "nbests" in job.result:
                print job.result["nbests"]
                if job.result["nbests"][0]["confidence"] > 0.15:
                    result = job.result["nbests"][0]["key"]
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

    def __init__(self, device, angus_root):
        super(Say, self).__init__(device)
        self.service = angus_root.services.get_service("text_to_speech")

    def call(self, text):
        if not text:
            return
        job = self.service.process({
            "text": text,
            "lang": "fr-FR",
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
            say.call(u"Désolé, peux-tu répéter ?")
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
    say.call("Pour qui veux-tu laisser le message ?")
    forwho = retry(4, who.call, timeout=10)
    if not forwho:
        return
    forwho = forwho.encode("utf-8")

    say.call("Tu peux laisser un message maintenant et le finir par stop.")
    wav_path = recorder.call(90)
    say.call("Je vais rejouer le message")
    play.call(wav_path)
    say.call("Le message est-il correcte ?")
    if retry(3, what.call, timeout=5) == "oui":
        say.call("Le message est maintenant sauvegardé pour %s" % forwho)
        MSGS.setdefault(forwho, []).append(Message(user, forwho, wav_path))

    say.call("Très bien, merci et à bientôt.")


def read_msgs(what, say, whois, who, play, msgs):
    while msgs:
        msg = msgs.pop()

        while True:
            say.call(
                "Ceci est un message déposé par %s à %s" %
                (msg.who.encode("utf-8"), msg.timestamp.strftime("%H:%M")))
            play.call(msg.wav)
            say.call("Veux-tu le réécouter ?")
            if retry(3, what.call, timeout=5) == "non":
                break

    say.call("Fin des nouveaux messages, merci.")


def main_behavior(what, say, whois, who, play):
    INTROS = [
        "Excusez-moi ?",
        "Venez me voir",
        "Pardon !",
        "Coucou !",
        "Je ne vous vois pas très bien",
    ]

    say.call(random.choice(INTROS))
    user = whois.call(timeout=30)
    if user:
        user = user.encode("utf-8")
        say.call("Bonjour %s !" % user)

        msgs = MSGS.get(user, [])

        if msgs:
            say.call(
                "Tu as %s nouveaux messages veux-tu les écouter ?" %
                (len(msgs)))
            if retry(3, what.call, timeout=5) == "oui":
                read_msgs(what, say, whois, who, play, msgs)
        else:
            say.call("Tu n'as pas de nouveau messages")

        say.call("Veux-tu laisser un message ?")

        if retry(3, what.call, timeout=5) == "oui":
            record_msgs(what, say, whois, who, play, user)
        else:
            say.call("Très bien bonne journée et à bientôt.")


if __name__ == "__main__":
    root = angus.connect()

    ########
    # Corpus
    ########
    album = {
        "Aurelien": ["ids/aurelien/face.jpg"],
        "Sylvain": ["ids/sylvain/face.jpg"],
    }

    targets = dict([(key, []) for key in album])

    vocabs = {
        "oui": ["vocabs/affirmatif-0.wav", "vocabs/affirmatif-1.wav", "vocabs/affirmatif-2.wav"],
        "non": ["vocabs/non-0.wav", "vocabs/non-1.wav", "vocabs/non-2.wav"],
    }

    stops = {"stop": [
        "vocabs/stop-0.wav",
        "vocabs/stop-1.wav",
        "vocabs/stop-2.wav",
        "vocabs/stop-3.wav"]
    }

    ##########
    # Services
    ##########
    audio_in = "HD Pro Webcam C920: USB Audio (hw:1,0)"
    audio_out = "convert"

    whois = Whois(0, root, album)
    say = Say(audio_out, root)
    what = What(audio_in, root, vocabs, service_version=2)
    recorder = RecordUntil(audio_in, root, stops, service_version=2)
    play = Play(audio_out)
    who = What(audio_in, root, targets, service_version=2)

    ###########
    # Main loop
    ###########

    say.call(
        "Bonjour, veuillez-bien vous placer devant la caméra pour que je puisse vous reconnaitre")

    while True:
        main_behavior(what, say, whois, who, play)
