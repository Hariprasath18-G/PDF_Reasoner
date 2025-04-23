import axios from 'axios';

const api = axios.create({
  baseURL: 'http://localhost:8000/api',
});

export const uploadPDF = async (files: File[]) => {
  const formData = new FormData();
  files.forEach(file => formData.append('files', file));
  return api.post('/upload_pdfs', formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  });
};

export const getPDF = async (pdfName: string) => api.get(`/get_pdf?pdfName=${pdfName}`, { responseType: 'blob' });

export const queryPDF = async (query: string) => api.post('/query', { query });

export const getAbstract = async () => api.post('/abstract');
export const getKeyFindings = async () => api.post('/key_findings');
export const getChallenges = async () => api.post('/challenges');