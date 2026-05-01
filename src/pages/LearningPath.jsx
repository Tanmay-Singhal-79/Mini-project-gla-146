import React, { useState } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { 
  CheckCircle2, 
  Circle, 
  PlayCircle, 
  Zap,
  Star,
  Map as MapIcon,
  Sparkles,
  Code2,
  Trophy,
  Clock,
  ChevronRight,
  BarChart3
} from 'lucide-react';
import { Button } from '../components/ui/Button';
import api from '../services/api';
import { syncProgress, getRobustProgress } from '../services/progressSync';

export const LearningPath = () => {
  const location = useLocation();
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [activeInterest, setActiveInterest] = useState(location.state?.interest || '');

  // 1. Fetch User Profile (to get default interest if none provided)
  const { data: profile } = useQuery({
    queryKey: ['profile'],
    queryFn: () => api.auth.me().then(res => res.data),
  });

  const effectiveInterest = activeInterest || profile?.current_interest;

  // 2. Fetch Learning Path Steps
  const { data: pathData, isLoading: pathLoading, isError } = useQuery({
    queryKey: ['path', effectiveInterest],
    queryFn: async () => {
        try {
            const fetchPromise = api.learning.generatePath(effectiveInterest).then(res => res.data.steps);
            const timeoutPromise = new Promise((_, reject) => setTimeout(() => reject(new Error("HyperTimeout")), 300));
            
            const steps = await Promise.race([fetchPromise, timeoutPromise]);
            if (steps && steps.length > 0) return steps;
            throw new Error("No data");
        } catch (err) {
            const { getRandomRoadmap } = await import('../data/mockRoadmaps');
            return getRandomRoadmap(effectiveInterest);
        }
    },
    enabled: !!effectiveInterest,
    staleTime: Infinity, // Keep the generated path persistent
  });

  // 3. Fetch Progress
  const { data: progressData } = useQuery({
    queryKey: ['progress'],
    queryFn: () => getRobustProgress(),
  });

  // 4. Fetch Favorites
  const { data: favoritesData } = useQuery({
    queryKey: ['favorites'],
    queryFn: () => api.progress.getFavorites().then(res => res.data),
  });

  const completeMutation = useMutation({
    mutationFn: (stepTitle) => syncProgress(stepTitle),
    onMutate: async (stepTitle) => {
      await queryClient.cancelQueries({ queryKey: ['progress'] });
      const previousProgress = queryClient.getQueryData(['progress']);
      queryClient.setQueryData(['progress'], (old) => {
        const completedItems = old?.completedItems || [];
        if (!completedItems.find(item => item.step_title === stepTitle)) {
          const newItems = [...completedItems, { step_title: stepTitle, status: 'completed', id: 'temp_' + Date.now() }];
          // Calculate new percentage based on 100 steps per domain (approx 500 total)
          const newPercentage = Math.round((newItems.length / 500) * 1000) / 10;
          return {
            ...old,
            completedItems: newItems,
            percentage: newPercentage
          };
        }
        return old;
      });
      return { previousProgress };
    },
    onError: (err, stepTitle, context) => {
      queryClient.setQueryData(['progress'], context.previousProgress);
      console.error("Critical Sync Failure:", err);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['progress'] });
    },
  });

  const favoriteMutation = useMutation({
    mutationFn: (stepTitle) => api.progress.toggleFavorite(stepTitle),
    onMutate: async (stepTitle) => {
      await queryClient.cancelQueries({ queryKey: ['favorites'] });
      const previousFavorites = queryClient.getQueryData(['favorites']);
      queryClient.setQueryData(['favorites'], (old) => {
        const favorites = old || [];
        const isFavorited = favorites.find(item => item.step_title?.trim() === stepTitle?.trim());
        if (isFavorited) {
          return favorites.filter(item => item.step_title?.trim() !== stepTitle?.trim());
        } else {
          return [...favorites, { step_title: stepTitle, id: 'temp_' + Date.now() }];
        }
      });
      return { previousFavorites };
    },
    onError: (err, stepTitle, context) => {
      queryClient.setQueryData(['favorites'], context.previousFavorites);
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['favorites'] });
    },
  });

  if (isError) {
    return (
      <div className="glass-card rounded-[3rem] p-16 text-center max-w-2xl mx-auto mt-20 border-red-100">
        <div className="w-24 h-24 bg-red-50 rounded-[2rem] flex items-center justify-center mx-auto mb-8">
          <Zap className="w-12 h-12 text-red-500" />
        </div>
        <h2 className="text-4xl font-black text-slate-800 mb-6">Neural Link Interrupted</h2>
        <p className="text-xl text-slate-500 font-medium mb-10">
          The AI engine encountered an anomaly: <br/>
          <span className="text-red-600 font-black">{error?.response?.data?.message || error?.message || "Internal Sequence Error"}</span>
        </p>
        <Button 
          onClick={() => queryClient.invalidateQueries({ queryKey: ['path'] })}
          className="btn-primary px-10 py-6 rounded-[1.5rem] text-lg font-black bg-red-600 hover:bg-red-700 shadow-red-200"
        >
          Retry Connection
        </Button>
      </div>
    );
  }

  if (!effectiveInterest) {
    return (
      <div className="glass-card rounded-[3rem] p-16 text-center max-w-2xl mx-auto mt-20">
        <div className="w-24 h-24 bg-emerald-50 rounded-[2rem] flex items-center justify-center mx-auto mb-8">
          <MapIcon className="w-12 h-12 text-primary" />
        </div>
        <h2 className="text-4xl font-black text-slate-800 mb-6">No Path Selected</h2>
        <p className="text-xl text-slate-500 font-medium mb-10">Start by choosing a domain or entering your interest in the Recommendations tab.</p>
        <Button 
          onClick={() => navigate('/recommendations')}
          className="btn-primary px-10 py-6 rounded-[1.5rem] text-lg font-black"
        >
          Choose a Domain
        </Button>
      </div>
    );
  }

  const steps = pathData || [];
  const completedTitles = new Set(progressData?.completedItems?.map(item => item.step_title?.trim()) || []);
  const favoriteTitles = new Set(favoritesData?.map(item => item.step_title?.trim()) || []);
  
  // Calculate percentage relative to CURRENT roadmap steps for better user feedback
  const currentCompletedCount = steps.filter(s => completedTitles.has(s.title?.trim())).length;
  const completionPercentage = steps.length > 0 ? Math.round((currentCompletedCount / steps.length) * 1000) / 10 : 0;

  return (
    <div className="space-y-12 pb-20">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8">
        <div>
          <div className="inline-flex items-center gap-3 px-5 py-2 rounded-full bg-primary text-white mb-6 shadow-lg shadow-primary/20">
            <Sparkles className="w-4 h-4" />
            <span className="text-xs font-black uppercase tracking-widest">AI Roadmap Generation</span>
          </div>
          <h1 className="text-6xl font-black text-slate-800 tracking-tighter mb-4">
            Path to <span className="text-primary underline decoration-emerald-100 decoration-8 underline-offset-8 italic">{effectiveInterest}</span>
          </h1>
          <p className="text-xl text-slate-500 font-medium max-w-2xl">
            Your personalized learning trajectory, optimized for rapid mastery and technical excellence.
          </p>
        </div>
        
        <div className="bg-white p-6 rounded-[2.5rem] border border-primary/10 shadow-xl shadow-emerald-900/5 flex items-center gap-6">
          <div className="text-right">
            <p className="text-xs font-black text-slate-400 uppercase tracking-widest mb-1">Total Progress</p>
            <p className="text-3xl font-black text-primary">{completionPercentage}%</p>
          </div>
          <div className="w-20 h-20 rounded-full border-[6px] border-emerald-50 border-t-primary flex items-center justify-center font-black text-primary relative">
            <Trophy className="w-8 h-8 opacity-20 absolute" />
            <span className="relative z-10 text-xl">{progressData?.completedItems?.length || 0}</span>
          </div>
        </div>
      </div>

      {/* Roadmap Visualization */}
      <div className="relative">
        <div className="absolute left-10 top-0 bottom-0 w-1 bg-gradient-to-b from-primary/20 via-primary/40 to-primary/20 rounded-full hidden md:block" />

        <div className="space-y-10">
          {steps.map((step, index) => {
            const isCompleted = completedTitles.has(step.title);
            const isFavorited = favoriteTitles.has(step.title);
            // Current is the first non-completed step
            const isCurrent = !isCompleted && steps.findIndex(s => !completedTitles.has(s.title)) === index;

            return (
              <div 
                key={step.id || index} 
                className={`relative pl-0 md:pl-24 group transition-all duration-500 ${isCurrent ? 'scale-[1.02]' : 'opacity-80 hover:opacity-100'}`}
              >
                {/* Node on Line */}
                <div className={`absolute left-[34px] top-10 w-4 h-4 rounded-full border-4 border-[var(--color-background)] z-10 transition-all duration-500 hidden md:block ${
                  isCompleted ? 'bg-primary scale-125' : isCurrent ? 'bg-primary animate-pulse ring-4 ring-primary/20' : 'bg-slate-300'
                }`} />

                <div className={`glass-card p-10 rounded-[3rem] relative overflow-hidden flex flex-col md:flex-row items-center gap-10 border-2 transition-all duration-500 ${
                  isCurrent ? 'border-primary/40 bg-white/60 shadow-[0_30px_60px_-15px_rgba(0,161,155,0.2)]' : 'border-transparent bg-white/40'
                }`}>
                  {/* Step Number with Toggle Completion */}
                  <button 
                    onClick={() => completeMutation.mutate(step.title)}
                    disabled={completeMutation.isPending}
                    className={`shrink-0 w-20 h-20 rounded-[2rem] flex items-center justify-center text-3xl font-black transition-all duration-500 relative group/check overflow-hidden ${
                      isCompleted ? 'bg-primary text-white shadow-lg shadow-primary/30' : 'bg-slate-50 text-slate-300 hover:bg-emerald-50 hover:text-primary'
                    }`}
                  >
                    <span className={`transition-opacity ${isCompleted ? 'opacity-0' : 'group-hover/check:opacity-0'}`}>{index + 1}</span>
                    <CheckCircle2 className={`absolute transition-opacity w-8 h-8 ${isCompleted ? 'opacity-100' : 'opacity-0 group-hover/check:opacity-100'}`} />
                  </button>

                  <div className="flex-1 text-center md:text-left">
                    <div className="flex items-center justify-center md:justify-start gap-3 mb-3">
                      <h3 className={`text-2xl font-black tracking-tight ${isCompleted ? 'text-slate-800' : 'text-slate-700'}`}>
                        {step.title}
                      </h3>
                      {isCompleted && <CheckCircle2 className="w-6 h-6 text-primary" />}
                      {isCurrent && <span className="bg-primary/10 text-primary text-[10px] font-black px-3 py-1 rounded-full uppercase tracking-widest">Active</span>}
                    </div>
                    <p className="text-slate-500 font-bold leading-relaxed mb-6">
                      {step.description}
                    </p>

                    <div className="flex flex-wrap gap-4 mb-6">
                        <span className={
                          step.difficulty === 'Beginner' ? 'badge-beginner' : 
                          step.difficulty === 'Intermediate' ? 'badge-intermediate' : 
                          'badge-advanced'
                        }>
                          <BarChart3 className="w-3 h-3 inline mr-1" />
                          {step.difficulty || 'Intermediate'}
                        </span>
                        <span className="flex items-center gap-1 text-slate-400 text-[10px] font-black uppercase tracking-widest bg-slate-50 px-3 py-1 rounded-full border border-slate-100">
                          <Clock className="w-3 h-3" />
                          {step.estimatedHours || 10} Hours
                        </span>
                      </div>

                      {step.milestones && (
                        <div className="mb-8 bg-slate-50/50 rounded-2xl p-6 border border-slate-100/50">
                          <p className="text-[10px] font-black text-slate-400 uppercase tracking-widest mb-3 flex items-center gap-2">
                            <ChevronRight className="w-3 h-3 text-primary" />
                            Key Milestones
                          </p>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-8 gap-y-2">
                            {step.milestones.map((milestone, idx) => (
                              <div key={idx} className="flex items-center gap-2 text-slate-600 text-sm font-bold">
                                <div className="w-1.5 h-1.5 rounded-full bg-primary/40" />
                                {milestone}
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                      
                      <div className="flex flex-wrap items-center justify-center md:justify-start gap-4">
                        <Button 
                          onClick={() => navigate(`/lesson/${encodeURIComponent(step.title)}`, { state: { steps } })}
                          className={`rounded-[1.2rem] px-8 py-4 font-black flex items-center gap-2 transition-all ${
                            isCurrent ? 'btn-primary' : 'bg-white text-slate-500 border border-slate-200 hover:border-primary/20 hover:text-primary shadow-sm'
                          }`}
                        >
                          <PlayCircle className="w-5 h-5" />
                          Master Concept
                        </Button>
                      </div>
                  </div>

                  {/* Star Button (Favorite Toggle) */}
                  <button 
                    onClick={() => favoriteMutation.mutate(step.title)}
                    className={`shrink-0 w-20 h-20 rounded-[2.5rem] flex items-center justify-center transition-all duration-500 border-2 ${
                      isFavorited 
                        ? 'bg-amber-50 border-amber-200 text-amber-500 scale-110 shadow-lg shadow-amber-200/50' 
                        : 'bg-slate-50/50 border-transparent text-slate-300 hover:bg-amber-50/50 hover:text-amber-400'
                    }`}
                  >
                    <Star className={`w-8 h-8 ${isFavorited ? 'fill-current' : ''}`} />
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
