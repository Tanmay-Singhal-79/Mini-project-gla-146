import React from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '../services/api';
import { Award, Calendar, TrendingUp, Zap, Target, Star, Brain, PlayCircle } from 'lucide-react';
import { Button } from '../components/ui/Button';
import { useNavigate } from 'react-router-dom';

export const Progress = () => {
  const navigate = useNavigate();
  
  const { data, isLoading } = useQuery({
    queryKey: ['progress'],
    queryFn: () => api.progress.get().then(res => res.data),
    refetchInterval: 10000,
  });

  const { data: favorites, isLoading: favoritesLoading } = useQuery({
    queryKey: ['favorites'],
    queryFn: () => api.progress.getFavorites().then(res => res.data),
  });

  return (
    <div className="space-y-16 pb-20 pt-10">
      {/* Header */}
      <section className="max-w-4xl">
        <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full bg-primary text-white mb-8 shadow-lg shadow-primary/20">
          <TrendingUp className="w-5 h-5" />
          <span className="text-xs font-black uppercase tracking-widest">Growth Metrics</span>
        </div>
        <h1 className="text-7xl font-black text-slate-800 tracking-tighter mb-6">
          Your Neural <span className="text-primary underline decoration-emerald-100 decoration-8 underline-offset-8 italic">Expansion</span>
        </h1>
        <p className="text-2xl text-slate-500 font-medium leading-relaxed">
          Tracking every synapse, every milestone, and every breakthrough in your journey to technical mastery.
        </p>
      </section>

      {/* Main Stats Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-10">
        <div className="lg:col-span-2 glass-card p-12 rounded-[4rem] bg-white/40 flex flex-col justify-between border-primary/5">
          <div className="flex items-center justify-between mb-16">
            <div>
              <h2 className="text-4xl font-black text-slate-800 mb-2">Overall Mastery</h2>
              <p className="text-slate-500 font-bold">Progress across active trajectories</p>
            </div>
            <div className="w-24 h-24 bg-primary rounded-[2rem] flex items-center justify-center shadow-2xl shadow-primary/20">
              <Target className="w-12 h-12 text-white" />
            </div>
          </div>
          
          <div className="space-y-8">
            <div className="flex items-end justify-between mb-2">
              <span className="text-8xl font-black text-primary tracking-tighter">
                {isLoading ? '...' : data?.percentage}%
              </span>
              <div className="text-right mb-4">
                <p className="text-xs font-black text-slate-400 uppercase tracking-widest">Next Milestone</p>
                <p className="text-xl font-black text-slate-800 italic">Advanced Architect</p>
              </div>
            </div>
            <div className="h-8 w-full bg-emerald-50 rounded-full overflow-hidden p-1.5 border border-primary/5 shadow-inner">
              <div 
                className="h-full bg-primary rounded-full shadow-[0_0_30px_rgba(0,161,155,0.4)] transition-all duration-1000 ease-out"
                style={{ width: `${data?.percentage || 0}%` }}
              />
            </div>
          </div>
        </div>

        <div className="glass-card p-12 rounded-[4rem] bg-primary text-white shadow-2xl shadow-primary/20 flex flex-col items-center justify-center text-center relative overflow-hidden">
          <div className="absolute top-0 right-0 -mr-20 -mt-20 opacity-10">
            <Award className="w-64 h-64" />
          </div>
          <div className="w-24 h-24 bg-white/10 backdrop-blur-xl rounded-[2.5rem] flex items-center justify-center mb-8 border border-white/20">
            <Star className="w-12 h-12 text-emerald-300" />
          </div>
          <h2 className="text-3xl font-black mb-4">Modules <br/> Completed</h2>
          <span className="text-9xl font-black tracking-tighter mb-6">
            {isLoading ? '0' : data?.completedItems?.length || 0}
          </span>
          <p className="text-emerald-50/60 font-bold uppercase tracking-widest text-sm">Industry Ready</p>
        </div>
      </div>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-16">
        {/* History / Milestones */}
        <section className="space-y-10">
          <div className="flex items-center justify-between">
            <h2 className="text-4xl font-black text-slate-800 tracking-tight">Milestones</h2>
            <div className="h-px flex-1 ml-10 bg-gradient-to-r from-primary/20 to-transparent" />
          </div>

          <div className="grid gap-6">
            {isLoading ? (
              [1, 2].map(i => <div key={i} className="h-24 glass-card rounded-[2rem] animate-pulse" />)
            ) : data?.completedItems?.length > 0 ? (
              data.completedItems.map((item, i) => (
                <div 
                  key={i} 
                  className="glass-card group p-8 rounded-[2.5rem] flex items-center justify-between bg-white/40 border-2 border-transparent hover:border-primary/10 transition-all duration-500"
                >
                  <div className="flex items-center gap-6">
                    <div className="w-16 h-16 bg-emerald-50 rounded-[1.5rem] flex items-center justify-center text-primary group-hover:bg-primary group-hover:text-white transition-all duration-500">
                      <Brain className="w-8 h-8" />
                    </div>
                    <div>
                      <h3 className="text-2xl font-black text-slate-800">{item.step_title}</h3>
                      <p className="text-slate-400 font-bold flex items-center gap-2">
                        <Calendar className="w-4 h-4" /> 
                        {item.completed_at ? new Date(item.completed_at).toLocaleDateString() : 'Recent Achievement'}
                      </p>
                    </div>
                  </div>
                  <div className="w-12 h-12 rounded-full bg-emerald-50 flex items-center justify-center text-primary">
                    <Zap className="w-6 h-6" />
                  </div>
                </div>
              ))
            ) : (
              <div className="text-center py-20 glass-card rounded-[3rem] border-dashed border-2 border-primary/10 bg-white/20">
                <p className="text-xl font-black text-slate-400 italic">No milestones registered yet.</p>
              </div>
            )}
          </div>
        </section>

        {/* Favorites */}
        <section className="space-y-10">
          <div className="flex items-center justify-between">
            <h2 className="text-4xl font-black text-slate-800 tracking-tight">Starred</h2>
            <div className="h-px flex-1 ml-10 bg-gradient-to-r from-primary/20 to-transparent" />
          </div>

          <div className="grid gap-6">
            {favoritesLoading ? (
              [1, 2].map(i => <div key={i} className="h-24 glass-card rounded-[2rem] animate-pulse" />)
            ) : favorites?.length > 0 ? (
              favorites.map((fav, i) => (
                <div 
                  key={i} 
                  className="glass-card group p-8 rounded-[2.5rem] flex items-center justify-between bg-white/40 border-2 border-transparent hover:border-primary/10 transition-all duration-500"
                >
                  <div className="flex items-center gap-6">
                    <div className="w-16 h-16 bg-amber-50 rounded-[1.5rem] flex items-center justify-center text-amber-500 group-hover:bg-amber-500 group-hover:text-white transition-all duration-500">
                      <Star className="w-8 h-8 fill-current" />
                    </div>
                    <h3 className="text-2xl font-black text-slate-800">{fav.step_title}</h3>
                  </div>
                  <Button 
                    variant="ghost" 
                    onClick={() => navigate(`/lesson/${encodeURIComponent(fav.step_title)}`)}
                    className="w-12 h-12 p-0 rounded-full bg-emerald-50 flex items-center justify-center text-primary hover:bg-primary hover:text-white transition-all"
                  >
                    <PlayCircle className="w-6 h-6" />
                  </Button>
                </div>
              ))
            ) : (
              <div className="text-center py-20 glass-card rounded-[3rem] border-dashed border-2 border-primary/10 bg-white/20">
                <p className="text-xl font-black text-slate-400 italic">No starred lessons yet.</p>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  );
};
