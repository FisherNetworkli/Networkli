declare module '../ml/models/content_recommender' {
  export class ContentRecommender {
    constructor();
    predict(content: string, interests: string[]): Promise<[number, boolean]>;
  }
} 