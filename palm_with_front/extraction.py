import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class User:
    code: int
    name: str
    profile_code: int
    position_fio: Optional[str] = None
    password: Optional[str] = None
    user_card: Optional[str] = None
    inn: Optional[str] = None

    def to_dict(self) -> dict:
        """Аналог Go-тегов json:"""
        res = {
            "code": self.code,
            "name": self.name,
            "profile-code": self.profile_code,
        }
        if self.position_fio:
            res["position-fio"] = self.position_fio
        if self.password:
            res["password"] = self.password
        if self.user_card:
            res["user-card"] = self.user_card
        if self.inn:
            res["inn"] = self.inn
        return res


class Extract:

    def __init__(self, logger: logging.Logger):
        self.log = logger
        self.users: List[User] = []

    def process_file(self, file_path: str):
        self.log.debug(f"extraction.ProcessFile: {file_path}")

        try:
            with open(file_path, "r", encoding="cp1251") as f:
                content = f.read()
        except Exception as e:
            raise e

        # Разделение по строкам, аналог strings.Split(..., "\n")
        # Отрезаем последний \r, если файл с Windows-переносами
        ss = [line.rstrip("\r") for line in content.split("\n")]

        if not ss or ss == [""]:
            raise Exception("empty extraction")

        for i, uu in enumerate(ss):
            if uu == "$$$DELETEALLUSERS":
                self.users = []
            elif uu == "$$$ADDUSERS":
                self.add_users(ss[i + 1 :])
            elif uu == "$$$DELETEUSERSBYCODE":
                self.del_user_by_code(ss[i + 1 :])

    def del_user_by_code(self, ss: List[str]):
        self.log.debug(f"extraction.DelUserByCode: {ss}")
        for code in ss:
            if code.startswith("$$$"):
                return
            if not code.strip():
                continue

            try:
                cc = int(code)
            except ValueError as e:
                raise ValueError(
                    f"wrong field Code({code}) in DelUser struct: {e}"
                )

            # Удаление элемента по коду
            self.users = [u for u in self.users if u.code != cc]

    def add_users(self, ss: List[str]):
        self.log.debug(f"extraction.AddUsers: {ss}")
        for i, uu in enumerate(ss):
            if not uu.strip():
                continue

            data = uu.split(";")

            if len(data) > 1:
                try:
                    code = int(data[0])
                except ValueError as e:
                    raise ValueError(
                        f"wrong field Code({uu}) in User[{i+1}] struct: {e}"
                    )
            else:
                if data[0].startswith("$$$") or data[0] == "":
                    return

            user = User(code=0, name="", profile_code=0)
            user.code = code

            if len(data) > 1:
                user.name = data[1]
            if len(data) > 2:
                user.position_fio = data[2]

            if len(data) > 3:
                try:
                    p_code = int(data[3])
                except ValueError as e:
                    raise ValueError(
                        f"wrong field Code({data[3]}) in User[{i+1}] struct: {e}"
                    )
                user.profile_code = p_code

            if len(data) > 4:
                user.password = data[4]
            if len(data) > 5:
                user.user_card = data[5]
            if len(data) > 6:
                user.inn = data[6]

            self.users.append(user)
            
class Extract:
    # ... (предыдущие методы: __init__, process_file и т.д.) ...

    def to_json(self, indent: int = None) -> str:
        """Конвертирует список распарсенных пользователей в JSON-строку.
        
        indent: если передать 4, JSON будет красиво отформатирован с отступами.
        """
        # Превращаем каждый объект User в словарь с учетом omitempty
        users_dicts = [user.to_dict() for user in self.users]
        
        # ensure_ascii=False сохраняет кириллицу (ФИО) в читаемом виде, а не в \u0445
        return json.dumps(users_dicts, ensure_ascii=False, indent=indent)