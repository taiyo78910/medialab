import datetime
import argparse
import re 

pars = argparse.ArgumentParser(description='Date')
pars.add_argument('date', type=str, default=None, nargs="?")
pars.add_argument('-d', '--dir',default=None)
pars.add_argument('-f','--from_d',default=None)
pars.add_argument('-t','--to_d',default=None)
args = pars.parse_args()

def main():
    global args

    try:
        date=get_date()
        d1 = datetime.datetime.strptime(date, '%Y-%m-%d_%H:%M:%S')
    except:
        exit()

    now = datetime.datetime.now()
    dt = now - d1

    del_flg=0   
    if (args.from_d is None) and (args.from_d is None):
        exit()
    else:
        if args.from_d is None:
            args.from_d=0   
        if args.to_d is None:
            args.to_d=1e30

    if ( dt.days >= int(args.from_d) ) and ( dt.days <= int(args.to_d) ):
        del_flg=1

    return_Keep_or_Delete( del_flg )

def get_date():
    global args
        if args.dir is not None:
                try:
                        return re.findall('\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}', args.dir)[0]
                except:
                        exit()
        else:
                return args.date

def return_Keep_or_Delete(i):
    if i==0:
        print("Keep")
    else:
        print("Delete")

main()