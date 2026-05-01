import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { MessageSquare, Send, User, Clock } from 'lucide-react';
import api from '../../services/api';
import { Button } from '../ui/Button';

export const CommentSection = ({ resourceId }) => {
  const [content, setContent] = useState('');
  const queryClient = useQueryClient();

  const { data: comments, isLoading } = useQuery({
    queryKey: ['comments', resourceId],
    queryFn: async () => {
      // Short delay for "Sync" feel
      await new Promise(resolve => setTimeout(resolve, 500));
      try {
        const res = await api.community.getComments(resourceId);
        if (res.data && res.data.length > 0) return res.data;
        throw new Error("No comments");
      } catch (err) {
        console.warn("Comment sync failed, using Social Cache:", err);
        const { mockComments } = await import('../../data/mockCommunity');
        return mockComments[resourceId] || [];
      }
    },
  });

  const addMutation = useMutation({
    mutationFn: (text) => api.community.addComment(resourceId, text),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['comments', resourceId] });
      queryClient.invalidateQueries({ queryKey: ['resources'] });
      setContent('');
    }
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    if (content.trim()) {
      addMutation.mutate(content.trim());
    }
  };

  return (
    <div className="mt-8 pt-8 border-t border-primary/5 space-y-8 animate-in fade-in slide-in-from-top-4 duration-500">
      <div className="flex items-center gap-3 mb-4">
        <MessageSquare className="w-5 h-5 text-primary" />
        <h4 className="text-sm font-black text-slate-800 uppercase tracking-widest">Discussions</h4>
      </div>

      <form onSubmit={handleSubmit} className="flex gap-4">
        <div className="flex-1 relative">
          <input 
            value={content}
            onChange={(e) => setContent(e.target.value)}
            placeholder="Contribute to the conversation..."
            className="w-full bg-white border border-primary/10 rounded-2xl py-4 px-6 text-sm font-bold text-slate-700 placeholder:text-slate-300 focus:border-primary/30 outline-none transition-all shadow-sm"
          />
        </div>
        <Button 
          type="submit"
          disabled={addMutation.isPending || !content.trim()}
          className="rounded-2xl px-6 py-4 bg-primary text-white shadow-lg shadow-primary/10"
        >
          <Send className="w-5 h-5" />
        </Button>
      </form>

      <div className="space-y-6">
        {isLoading ? (
          <div className="h-20 bg-slate-50 animate-pulse rounded-2xl" />
        ) : comments?.length > 0 ? (
          comments.map((comment) => (
            <div key={comment.id} className="flex gap-5 group">
              <div className="shrink-0 w-10 h-10 rounded-xl bg-emerald-50 flex items-center justify-center text-primary group-hover:scale-110 transition-transform">
                <User className="w-5 h-5" />
              </div>
              <div className="flex-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-sm font-black text-slate-800">{comment.creator_name}</span>
                  <span className="text-[10px] font-black text-slate-300 uppercase tracking-widest flex items-center gap-1">
                    <Clock className="w-3 h-3" />
                    {new Date(comment.created_at).toLocaleDateString()}
                  </span>
                </div>
                <p className="text-sm text-slate-500 font-bold leading-relaxed">
                  {comment.content}
                </p>
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-10 bg-slate-50/50 rounded-3xl border-2 border-dashed border-primary/5">
            <p className="text-xs font-black text-slate-300 uppercase tracking-widest">No transmissions yet. Start the thread.</p>
          </div>
        )}
      </div>
    </div>
  );
};
