import init_ea
import ea

if __name__ == '__main__':
    con, cur = init_ea.connect_db()
    status = init_ea.read_status(cur)

    print("\nStart EA Process")
    ea.ea_loop(con, cur, status)

    init_ea.close_db(con, cur)
