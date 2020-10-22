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
    # construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default="127.0.0.1", help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=5000, help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)
