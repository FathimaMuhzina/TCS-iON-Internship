import pandas as pd
from flask import Flask,request,render_template
import pickle
from sklearn.preprocessing import StandardScaler
model = pickle.load(open('model.pkl', 'rb'))


app = Flask(__name__)

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/show')
def show():
    return render_template('prediction.html')

@app.route('/prediction', methods=['GET','POST'])
def prediction():
    if(request.method == 'POST'):
        power=float(request.form['power'])
        blue=float(request.form['blue'])
        tg=float(request.form['tg'])
        clock=float(request.form['clock'])
        dual=float(request.form['dual'])
        ts=float(request.form['ts'])
        fc=float(request.form['fc'])
        fg=float(request.form['fg'])
        wf=float(request.form['wf'])
        ncores=float(request.form['ncores'])
        mdep=float(request.form['mdep'])
        mwt=float(request.form['mwt'])
        pc=float(request.form['pc'])
        pht=float(request.form['pht'])
        pwt=float(request.form['pwt'])
        ram=float(request.form['ram'])
        sht=float(request.form['sht'])
        swt=float(request.form['swt'])
        mem=float(request.form['mem'])
        talk=float(request.form['talk'])
    
        data = {'battery_power':[power], 'blue':[blue], 'clock_speed':[clock], 'dual_sim':[dual], 'fc':[fc], 'four_g':[fg], 'int_memory':[mem], 'm_dep':[mdep], 'mobile_wt':[mwt], 'n_cores':[ncores], 'pc':[pc], 'px_height':[pht], 'px_width':[pwt], 'ram':[ram], 'sc_h':[sht], 'sc_w':[swt], 'talk_time':[talk], 'three_g':[tg], 'touch_screen':[ts], 'wifi':[wf]}
  
       
        df = pd.DataFrame(data)
        
        scaled=StandardScaler()
        df_scaled = pd.DataFrame(scaled.fit_transform(df),columns = df.columns)
      
        pred = model.predict(df_scaled)
        print(pred)
    
        return render_template ('result.html',prediction_text="Your phone has a price range of {} ".format(pred))

if __name__=='__main__':
    app.run(port=8000)