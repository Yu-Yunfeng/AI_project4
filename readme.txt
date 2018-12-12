{\rtf1\ansi\ansicpg1252\cocoartf1671\cocoasubrtf100
{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\margl1440\margr1440\vieww10800\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 \'931\'94 represents training mode and \'932\'94 represents test mode\
\
python3 dacision_tree.py iris.data.discrete.txt 1\
python3 dacision_tree.py iris.data.discrete.txt 2\
python3 dacision_tree.py scale1.data.txt 1\
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0
\cf0 python3 dacision_tree.py scale1.data.txt 2\
\
\'930.2\'94 represents the test size and \'9310\'94 represents the training times\
python3 validation.py iris.data.txt 0.2 10\
python3 validation.py scale1.data.txt 0.2 10\
\
}