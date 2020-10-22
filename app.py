import argparse

from flask import Flask, render_template, Response

from face_recognition import MaskDetection

# initialize the mask detector and the total number of frames
md = MaskDetection()
app = Flask(__name__)


@app.route('/')
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/video_feed')
def video_feed():
    # return the response generated along with the specific media type (mime type)
    # start a thread that will perform motion detection
    """
    t = threading.Thread(target=md.detect_mask())
    t.daemon = True
    t.start()
    """
    return Response(md.generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # start the flask app
    app.run(debug=True, threaded=True, use_reloader=False)
