import os
from houseprices import app

if __name__ == '__main__':
    try:
        port = os.environ['PORT']
    except KeyError:
        port = 5000

    app.run(host="0.0.0.0", port=int(port), debug=True, use_reloader=False)