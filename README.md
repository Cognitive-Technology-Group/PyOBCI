Python Library for OpenBCI
==============

![alt tag](https://raw.github.com/theRealWardo/Python_OpenBCI/master/architecture.png)

- **open_bci.py** manages the connection between the OpenBCI board and Python
- **udp_server.py** exposes the data over UDP
- **socket_server.js** a Node.js server that retransmits the data over a Web Socket
- **htdocs/index.html** a hack to display data using D3.js

Running the Server
--------------

- Plugin the board
- `python udp_server.py --json` (add the `--filter_data` command to enable the band-stop filter on the board)
- Optionally use `python udp_client.py --json` to verify data is coming through
- Run `npm install` to make sure you have the dependencies
- Run `node socket_server.js`
- Visit [http://127.0.0.1:8880](http://127.0.0.1:8880) to see your brain waves
- Optionally use `python socket_client.py` to view the Web Socket data coming back into Python (requires socketio-client)

Running the Python server/client without the --json flag will cause the OpenBCISample object to be used as the data transmission mechanism. This is for people that want to do some processing in Python.

Dependency List
--------------

Python UDP demos require:
- pyserial

Node sample requires:
- socket.io

Python Web Socket requires:
- socketio-client
