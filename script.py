from pydub import playback
import pydub

path = "audio/geo-tagging.mp3"
sound = pydub.AudioSegment.from_file("audio/greatest-local-seo.mp3", format="mp3")
playback.play(sound) 