#!/usr/bin/bash
python3 -m venv venv
echo "a() {" > rc
echo '    source "$(pwd)/venv/bin/activate"' >> rc

cat >> rc << EOF
}
da() {
    deactivate
}
EOF

source rc
a
python3 -m pip install -r requirements.txt
