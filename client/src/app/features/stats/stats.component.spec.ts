import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { ApiService } from '../../core/services/api.service';
import { I18nService } from '../../core/services/i18n.service';
import { StatsComponent } from './stats.component';

describe('StatsComponent', () => {
  let component: StatsComponent;
  let mockApi: { get: jest.Mock };

  const mockOverview = {
    total_photos: 1000,
    total_persons: 50,
    avg_score: 6.5,
    avg_aesthetic: 7.2,
    avg_composition: 5.8,
    total_faces: 300,
    total_tags: 150,
    date_range_start: '2023-01-01',
    date_range_end: '2024-12-31',
  };

  const mockGear = {
    cameras: [{ name: 'Canon R5', count: 500, avg_aggregate: 7.1, avg_aesthetic: 7.5 }],
    lenses: [{ name: 'RF 50mm', count: 200, avg_aggregate: 7.5, avg_aesthetic: 7.3 }],
    combos: [{ name: 'Canon R5 + RF 50mm', count: 100, avg_aggregate: 7.2 }],
    categories: [],
  };

  const mockCategories = [
    { category: 'portrait', count: 300, percentage: 0.3, avg_score: 7.0 },
    { category: 'landscape', count: 200, percentage: 0.2, avg_score: 6.5 },
  ];

  const mockScoreBins = [
    { range: '0-1', min: 0, max: 1, count: 10, percentage: 0.01 },
    { range: '9-10', min: 9, max: 10, count: 50, percentage: 0.05 },
  ];

  const mockTimeline = [
    { period: '2024-01', count: 100, avg_score: 6.8 },
    { period: '2024-02', count: 120, avg_score: 7.0 },
  ];

  const mockTopCameras = [
    { name: 'Canon R5', count: 500, avg_score: 7.1, avg_aesthetic: 7.5 },
  ];

  const mockCorrData = {
    labels: ['100', '200', '400'],
    metrics: { aggregate: [6.5, 7.0, 6.8] },
    x_axis: 'iso',
    group_by: '',
  };

  /**
   * Create component with api.get mock pre-configured.
   * The constructor calls loadAll() immediately, so the mock must be ready.
   */
  /** Returns safe empty data for paths that feed into effects expecting arrays */
  function safeDefault(path: string) {
    if (path === '/stats/categories') return of([]);
    if (path === '/stats/score_distribution') return of([]);
    if (path === '/stats/timeline') return of([]);
    if (path === '/stats/top_cameras') return of([]);
    if (path === '/stats/gear') return of({ cameras: [], lenses: [], combos: [], categories: [] });
    return of({});
  }

  function createComponent(getMock?: jest.Mock): StatsComponent {
    mockApi = {
      get: getMock ?? jest.fn((path: string) => safeDefault(path)),
    };

    TestBed.configureTestingModule({
      providers: [
        { provide: ApiService, useValue: mockApi },
        { provide: I18nService, useValue: { t: (key: string) => key } },
      ],
    });
    return TestBed.runInInjectionContext(() => new StatsComponent());
  }

  afterEach(() => {
    TestBed.resetTestingModule();
  });

  describe('loadAll()', () => {
    it('should fetch overview and set the signal', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/overview') return of(mockOverview);
        return safeDefault(path);
      });
      component = createComponent(getMock);

      // Wait for constructor's loadAll to complete
      await flushPromises();

      expect(mockApi.get).toHaveBeenCalledWith('/stats/overview', expect.any(Object));
      expect(component.overview()).toEqual(mockOverview);
      expect(component.loading()).toBe(false);
    });

    it('should set loading to false even when overview fails', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/overview') return throwError(() => new Error('fail'));
        return safeDefault(path);
      });
      component = createComponent(getMock);

      await flushPromises();

      expect(component.loading()).toBe(false);
      expect(component.overview()).toBeNull();
    });

    it('should kick off parallel loads after overview', async () => {
      const getMock = jest.fn((path: string) => safeDefault(path));
      component = createComponent(getMock);

      await flushPromises();

      expect(mockApi.get).toHaveBeenCalledWith('/stats/overview', expect.any(Object));
      expect(mockApi.get).toHaveBeenCalledWith('/stats/gear', expect.any(Object));
      expect(mockApi.get).toHaveBeenCalledWith('/stats/categories', expect.any(Object));
      expect(mockApi.get).toHaveBeenCalledWith('/stats/score_distribution', expect.any(Object));
      expect(mockApi.get).toHaveBeenCalledWith('/stats/timeline', expect.any(Object));
      expect(mockApi.get).toHaveBeenCalledWith('/stats/top_cameras', expect.any(Object));
    });
  });

  describe('loadGear()', () => {
    it('should fetch gear stats and set cameras/lenses/combos signals', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/gear') return of(mockGear);
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      expect(component.cameras()).toEqual([{ name: 'Canon R5', count: 500, avg_score: 7.1, avg_aesthetic: 7.5 }]);
      expect(component.lenses()).toEqual([{ name: 'RF 50mm', count: 200, avg_score: 7.5, avg_aesthetic: 7.3 }]);
      expect(component.combos()).toEqual([{ name: 'Canon R5 + RF 50mm', count: 100, avg_score: 7.2 }]);
      expect(component.gearLoading()).toBe(false);
    });

    it('should set gearLoading to false on error', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/gear') return throwError(() => new Error('fail'));
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      expect(component.gearLoading()).toBe(false);
    });
  });

  describe('loadCategories()', () => {
    it('should fetch categories and set the signal', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/categories') return of(mockCategories);
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      expect(component.categoryStats()).toEqual(mockCategories);
      expect(component.categoriesLoading()).toBe(false);
    });
  });

  describe('loadScoreDistribution()', () => {
    it('should fetch score bins and set the signal', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/score_distribution') return of(mockScoreBins);
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      expect(component.scoreBins()).toEqual(mockScoreBins);
      expect(component.scoreLoading()).toBe(false);
    });
  });

  describe('loadTimeline()', () => {
    it('should fetch timeline and set the signal', async () => {
      const monthly = [
        { month: '2024-01', count: 100, avg_score: 6.8 },
        { month: '2024-02', count: 120, avg_score: 7.0 },
      ];
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/timeline') return of({ monthly });
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      expect(component.timeline()).toEqual(mockTimeline);
      expect(component.timelineLoading()).toBe(false);
    });
  });

  describe('loadTopCameras()', () => {
    it('should fetch top cameras and set the signal', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/top_cameras') return of(mockTopCameras);
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      expect(component.topCameras()).toEqual(mockTopCameras);
    });
  });

  describe('loadCorrelation()', () => {
    it('should fetch correlations and set corrData signal', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/correlations') return of(mockCorrData);
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      // loadCorrelation is on-demand, not called by loadAll
      await component.loadCorrelation();

      expect(mockApi.get).toHaveBeenCalledWith('/stats/correlations', expect.any(Object));
      expect(component.corrData()).toEqual(mockCorrData);
      expect(component.correlationLoading()).toBe(false);
    });

    it('should set correlationLoading to false on error', async () => {
      const getMock = jest.fn((path: string) => {
        if (path === '/stats/correlations') return throwError(() => new Error('fail'));
        return safeDefault(path);
      });
      component = createComponent(getMock);
      await flushPromises();

      await component.loadCorrelation();

      expect(component.correlationLoading()).toBe(false);
      expect(component.corrData()).toBeNull();
    });
  });
});

function flushPromises(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}
