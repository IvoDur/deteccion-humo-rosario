import sqlite3
import datetime

class Database:
    def __init__(self, db):
        self.conn = sqlite3.connect(db)
        self.cur = self.conn.cursor()
        self.cur.execute("""
                CREATE TABLE IF NOT EXISTS  fire_detections ( 
                        id INTEGER PRIMARY KEY AUTOINCREMENT, 
                        time datetime DEFAULT CURRENT_TIMESTAMP, 
                        frame blob
                        )""")
        self.conn.commit()

    def can_save(self):
        last = self.get_last()
        if last:
            # convirto str a datetime
            last_time = datetime.datetime.strptime(last[0][1], '%Y-%m-%d %H:%M:%S')
            print( f"Pasaron {(datetime.datetime.now() - last_time).total_seconds()}")
            if (datetime.datetime.now() - last_time).total_seconds() < 60:
                print("Todavia no")
                return False
            else:
                print("Puedo guardar")
                return True
        else:
            return True

    def insert(self, frame, time=None):
        frame = sqlite3.Binary(frame)
        if self.can_save():
            if time:
                self.cur.execute("INSERT INTO fire_detections (time,frame) VALUES  (?, ?)", (time, frame))
            else:
                tiempo = datetime.datetime.now()
                tiempo = f"{tiempo.year}-{tiempo.month}-{tiempo.day} {tiempo.hour}:{tiempo.minute}:{tiempo.second}"
                print(tiempo)
                self.cur.execute("INSERT INTO fire_detections (time,frame) VALUES  (?, ?)", (tiempo, frame))
                # self.cur.execute("INSERT INTO fire_detections (frame) VALUES  (?)", (frame,))
            self.conn.commit()

    def view(self):
        self.cur.execute("SELECT * FROM fire_detections")
        rows = self.cur.fetchall()
        return rows

    def get_last(self):
        self.cur.execute("SELECT * FROM fire_detections ORDER BY id DESC LIMIT 1")
        rows = self.cur.fetchall()
        return rows
 

    def __del__(self):
        self.conn.close()


db = Database("fire_detections.db")
for algo in db.view():
    # if algo[0] == 1:
    print(f"{algo[0]}: T:{algo[1]}")
        # print(algo[2])
        # print(algo.id)
    # print(algo)
    # pass


# last = db.get_last()
# ## si el ultimo se guardo hace menos de 1 minuto no guardamos
# if last:
#     if (datetime.datetime.now() - last[0][1]).total_seconds() < 60:
#         print("no guardamos")
#         continue
#     else:
#         print("guardamos")
#         db.insert(frame, datetime.datetime.now())

