import threading
from flask import Flask, request, jsonify
from flask_socketio import SocketIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins='*')
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

_thread     = None
_stop_event = threading.Event()
_is_training = False


def _emit(event, data):
    socketio.emit(event, data)


@socketio.on('connect')
def on_connect():
    socketio.emit('status', {'training': _is_training})


@app.route('/api/status')
def status():
    return jsonify({'training': _is_training})


@app.route('/api/start', methods=['POST'])
def start_training():
    global _thread, _stop_event, _is_training
    if _is_training:
        return jsonify({'error': 'Already training'}), 400

    config = request.get_json(force=True) or {}
    _stop_event = threading.Event()
    _is_training = True

    def run():
        global _is_training
        try:
            from snake import train
            train(config, _stop_event, _emit)
        except Exception as exc:
            _emit('training_error', {'message': str(exc)})
        finally:
            _is_training = False

    _thread = threading.Thread(target=run, daemon=True)
    _thread.start()
    return jsonify({'status': 'started'})


@app.route('/api/stop', methods=['POST'])
def stop_training():
    _stop_event.set()
    return jsonify({'status': 'stopping'})


if __name__ == '__main__':
    print('Snake DQN backend  ->  http://localhost:5000')
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)
