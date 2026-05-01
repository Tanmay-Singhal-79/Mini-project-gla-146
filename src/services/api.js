import axios from 'axios';
import { auth } from '../firebase';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const apiClient = axios.create({
  baseURL: API_URL,
  timeout: 60000, // 60 seconds
  headers: {
    'Content-Type': 'application/json',
  },
});

// ─── Request Interceptor: Always get a FRESH Firebase token ──────────────────
// Never read from localStorage — Firebase auto-refreshes tokens transparently.
apiClient.interceptors.request.use(
  async (config) => {
    const currentUser = auth.currentUser;
    if (currentUser) {
      // getIdToken() returns cached token if valid, auto-refreshes if expired
      const token = await currentUser.getIdToken();
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ─── Response Interceptor: Handle 401 (token expiry) ─────────────────────────
// We export a setup function so AuthContext can inject the logout callback.
export const setupAxiosInterceptors = (logout) => {
  apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
      if (error.response?.status === 401) {
        logout();
      }
      return Promise.reject(error);
    }
  );
};

// ─── Structured API Service (real HTTP calls) ─────────────────────────────────
export const api = {
  auth: {
    me: () =>
      apiClient.get('/me'),

    login: (credentials) =>
      apiClient.post('/login', credentials),

    signup: (userData) =>
      apiClient.post('/signup', userData),
  },

  learning: {
    getRecommendations: (interest) =>
      apiClient.post('/recommend', { interest }),

    generatePath: (interest = 'Web Development') =>
      apiClient.post('/generate-path', { interest }),

    getLesson: (step_title) =>
      apiClient.get(`/lesson/${encodeURIComponent(step_title)}`),
  },

  progress: {
    get: () =>
      apiClient.get('/progress'),

    update: (step_title) =>
      apiClient.post('/progress', { step_title }),

    getFavorites: () =>
      apiClient.get('/favorites'),

    toggleFavorite: (step_title) =>
      apiClient.post('/favorites', { step_title }),
  },

  community: {
    getResources: () =>
      apiClient.get('/resource'),

    addResource: (title) =>
      apiClient.post('/resource', { title }),

    upvote: (resource_id) =>
      apiClient.post('/upvote', { resource_id }),

    getComments: (resource_id) =>
      apiClient.get(`/resource/${resource_id}/comments`),

    addComment: (resource_id, content) =>
      apiClient.post(`/resource/${resource_id}/comments`, { content }),
  },
};

export default api;
