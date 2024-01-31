"""
Script is directly inspired by the work of SzieberthAdam and his code:
https://github.com/FUMBBLPlus/fumbblreplay/tree/master

"""
import fumbblreplay.fumbblreplay.fetcher as fumbbl
import json
import os

from pathlib import Path

import xml.etree.ElementTree as ET
import lzstring

JSON_SEPARATORS = ",", ":"
LZString = lzstring.LZString()


def get_files(path=os.getcwd()):
    paths = []
    close_session = None
    replay = None

    for file in os.listdir(path):
        if file.endswith(".jnlp"):
            paths.append(os.path.join(path, file))
        if file == "clientCloseSession.json":
            with Path(os.path.join(path, file)).open() as f:
                close_session = json.load(f)
        if file == "clientReplay.json":
            with Path(os.path.join(path, file)).open() as f:
                replay = json.load(f)

    return paths, close_session, replay


def get_id(file_content):
    root = ET.fromstring(file_content)
    arg_arr = []
    for argument in root.iter("argument"):
        if len(arg_arr) > 0:
            game_id = argument.text
            break
        if argument.text == "-gameId":
            arg_arr.append(argument.text)
    try:
        return game_id
    except Exception as e:
        print("No game id found in the file")


if __name__ == '__main__':
    # Todo: Add more reliable way of converting games to json
    file_paths, close_session, replay = get_files()
    for file_path in file_paths:
        with open(file_path, "r") as file:
            file_content = file.read()
        game_id = get_id(file_content)
        test = fumbbl.get_replay_data(game_id)
        file_name = "data" + str(game_id) + ".json"
        with open(file_name, "wt") as file:
            json.dump(test, file, indent=4)
        print("Written output to:", file_name)


