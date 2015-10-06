import Queue
import StringIO
import time
import wave
import pyaudio


class AudioService(object):
    """
    Generic class to handle sound.
    """

    def __init__(self, device):
        """
        Create a new audio service that use the device with the given name
        """
        # PyAudio is a portaudio binding to manage sound
        self.audio = pyaudio.PyAudio()

        # Save default sample rate to prevent from forcing an other one
        try:
            self.device = device
            desc = self.audio.get_device_info_by_index(device)
            self.rate = int(desc["defaultSampleRate"])
        except Exception as e:
            # The device was not found
            raise Exception("No such device: %s" % device)

    def __del__(self):
        # Clean destroy of audio object
        self.audio.terminate()

    def record(self, timeout, callback):
        """
        Open an audio stream and call the callback every 500ms with the last
        piece of sound (in wav format).
        """

        # queue is an accumaltor that store sound chunks
        queue = Queue.Queue()

        # pyaudio can register a callback to work on sound asynchronously,
        # the callback just store the sound chunk in the accumulator
        def audio_callback(in_data, frame_count, time_info, status):
            queue.put(in_data)
            return (in_data, pyaudio.paContinue)

        # create a stream audio with pyaudio with default parameters
        # and the previous callback
        stream = self.audio.open(format=pyaudio.paInt16,
                                 channels=1,
                                 rate=self.rate,
                                 input=True,
                                 frames_per_buffer=8192,
                                 input_device_index=self.device,
                                 stream_callback=audio_callback)

        # start timestamp
        start = time.time()

        result = None
        while (time.time() - start) < timeout:
            # evry 500ms
            time.sleep(0.5)
            # check accumulator size, if no new chunk, loop
            if queue.qsize() == 0:
                continue

            # do not stack too much chunk
            while queue.qsize() > 3:
                queue.get()
            # if some chunks are available
            # create a in-memory file
            buff = StringIO.StringIO()

            # initialize a wave format
            wf = wave.open(buff, 'w')
            wf.setnchannels(1)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)

            # and write all chunks in memory in wav format
            while queue.qsize() > 0:
                wf.writeframes(queue.get())

            wf.close()

            # call the callback with the in-memory wav file
            result = callback(buff.getvalue())

            # if callback have a result, stop now
            if result:
                break

        # clean stream and return the result
        stream.close()
        return result

    def play(self, f):
        """
        Play wav buffer on the device synchronously
        """

        # define a pyaudio callback to read frames from the wav structure
        def callback(in_data, frame_count, time_info, status):
            data = f.readframes(frame_count)
            return (data, pyaudio.paContinue)

        # initialize audio stream for output with previous callback
        stream = self.audio.open(format=self.audio.get_format_from_width(f.getsampwidth()),
                                 channels=f.getnchannels(),
                                 rate=f.getframerate(),
                                 output_device_index=self.device,
                                 output=True,
                                 stream_callback=callback)

        # wait the end of audio streaming
        while stream.is_active():
            time.sleep(0.1)

        # clean stream
        stream.close()
