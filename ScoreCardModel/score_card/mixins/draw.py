#coding:utf-8
import matplotlib.pyplot as plt
from sklearn import metrics
class DrawMixin(object):
    """需要self.get_ks,self.get_scores,self.tag"""
    def drawks(self,b=100,o=1,p=20,n=100):
        kss = self.get_ks(b=b,o=o,p=p,n=n)
        X = [i for i,_,_,_,_,_ in kss]
        Y_good = [i for _,i,_,_,_,_ in kss]
        Y_bad = [i for _,_,i,_,_,_ in kss]
        Y_ks = [i for _,_,_,i,_,_ in kss]
        Y_good_bad = [i for _,_,_,_,i,_ in kss]
        plt.figure(figsize=(8,5), dpi=80)#设置图片大小和dpi
        plt.subplot(111)
        ax = plt.axes([0.025,0.025,0.95,0.95])
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.01))
        ax.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
        ax.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
        ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
        ax.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
        plt.plot(X,Y_good,color="blue",label="good rat")
        plt.plot(X,Y_bad,color="red",label="bad rat")
        plt.plot(X,Y_ks,color="yellow",label="KS")
        plt.plot(X,Y_good_bad,color="green",label="good_bad")
        #plt.plot(X,Y_roc,color="pink",label="roc_auc_score")
        t1 = 0.05
        plt.plot([t1,t1],[0,Y_good_bad[5]], color ='brown', linewidth=1.5, linestyle="--")
        plt.annotate(str(round(Y_good_bad[5],4)),xy=(t1,Y_good_bad[5]), xycoords='data',
            xytext=(70, 60), textcoords='offset points', fontsize=14,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        t2 = 0.08
        plt.plot([t2,t2],[0,Y_good_bad[8]], color ='brown', linewidth=1.5, linestyle="--")
        plt.annotate(str(round(Y_good_bad[8],4)),xy=(t2,Y_good_bad[8]), xycoords='data',
            xytext=(90, 50), textcoords='offset points', fontsize=14,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        t3 = 0.1
        plt.plot([t3,t3],[0,Y_good_bad[10]], color ='brown', linewidth=1.5, linestyle="--")
        plt.annotate(str(round(Y_good_bad[10],4)),xy=(t3,Y_good_bad[9]), xycoords='data',
            xytext=(120, 40), textcoords='offset points', fontsize=14,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
        plt.legend(loc='upper center')
        plt.show()

    def drawroc(self,b=100,o=1,p=20,n=100):
        scores = [score for score,_ in self.get_scores(b=b,o=o,p=p)]
        Y = [int(i)+1 for i in self.tag]

        fpr, tpr, thresholds = metrics.roc_curve(Y, scores, pos_label=2)

        plt.figure(figsize=(8,5), dpi=80)#设置图片大小和dpi
        plt.subplot(111)
        plt.plot(fpr,tpr,color="blue",label="orc")
        plt.legend(loc='upper center')
        plt.show()
