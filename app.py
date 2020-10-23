from flask import Flask, render_template, Response

from mask_detection import MaskDetection

# initialize the mask detector
md = MaskDetection()
app = Flask(__name__)


@app.route('/')
def index(name=None):
    return render_template('index.html', name=name)


@app.route('/video_feed')
def video_feed():
    # return the response generated along with the specific media type (mime type)
    return Response(md.generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == '__main__':
    # start the flask app
    app.run(debug=True)
