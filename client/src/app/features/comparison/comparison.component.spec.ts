import { TestBed } from '@angular/core/testing';
import { of, throwError } from 'rxjs';
import { MatSnackBar } from '@angular/material/snack-bar';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { I18nService } from '../../core/services/i18n.service';
import { ComparisonComponent, WeightIconPipe } from './comparison.component';

describe('ComparisonComponent', () => {
  let component: ComparisonComponent;
  let mockApi: { get: jest.Mock; post: jest.Mock; delete: jest.Mock };
  let mockSnackBar: { open: jest.Mock };
  let mockI18n: { t: jest.Mock };
  let mockAuth: { isEdition: jest.Mock };

  beforeEach(() => {
    mockApi = {
      get: jest.fn(() => of({ categories: [] })),
      post: jest.fn(() => of({})),
      delete: jest.fn(() => of({})),
    };
    mockSnackBar = { open: jest.fn() };
    mockI18n = { t: jest.fn((key: string) => key) };
    mockAuth = { isEdition: jest.fn(() => true) };

    TestBed.configureTestingModule({
      providers: [
        ComparisonComponent,
        { provide: ApiService, useValue: mockApi },
        { provide: MatSnackBar, useValue: mockSnackBar },
        { provide: I18nService, useValue: mockI18n },
        { provide: AuthService, useValue: mockAuth },
      ],
    });
    component = TestBed.inject(ComparisonComponent);
  });

  describe('WeightIconPipe', () => {
    let pipe: WeightIconPipe;

    beforeEach(() => {
      pipe = new WeightIconPipe();
    });

    it('should return correct icon for known keys', () => {
      expect(pipe.transform('aesthetic_percent')).toBe('auto_awesome');
      expect(pipe.transform('composition_percent')).toBe('grid_on');
      expect(pipe.transform('face_quality_percent')).toBe('face');
      expect(pipe.transform('tech_sharpness_percent')).toBe('center_focus_strong');
      expect(pipe.transform('color_percent')).toBe('palette');
      expect(pipe.transform('exposure_percent')).toBe('exposure');
      expect(pipe.transform('noise_percent')).toBe('grain');
    });

    it('should return fallback icon for unknown keys', () => {
      expect(pipe.transform('unknown_key')).toBe('tune');
    });
  });

  describe('setWeight', () => {
    it('should update a weight value', () => {
      component.weights.set({ aesthetic_percent: 30, composition_percent: 20 });

      component.setWeight('aesthetic_percent', 45);

      expect(component.weights()['aesthetic_percent']).toBe(45);
    });

    it('should preserve other weights', () => {
      component.weights.set({ aesthetic_percent: 30, composition_percent: 20 });

      component.setWeight('aesthetic_percent', 45);

      expect(component.weights()['composition_percent']).toBe(20);
    });

    it('should add new weight keys', () => {
      component.weights.set({});

      component.setWeight('noise_percent', 10);

      expect(component.weights()['noise_percent']).toBe(10);
    });
  });

  describe('weightTotal computed', () => {
    it('should sum all weight values', () => {
      component.weights.set({
        aesthetic_percent: 35,
        composition_percent: 25,
        face_quality_percent: 20,
        tech_sharpness_percent: 10,
        color_percent: 5,
        exposure_percent: 3,
        noise_percent: 2,
      });

      expect(component.weightTotal()).toBe(100);
    });

    it('should return 0 for empty weights', () => {
      component.weights.set({});
      expect(component.weightTotal()).toBe(0);
    });

    it('should handle partial weights', () => {
      component.weights.set({ aesthetic_percent: 30, composition_percent: 20 });
      expect(component.weightTotal()).toBe(50);
    });
  });

  describe('loadCategories', () => {
    it('should fetch categories from API', async () => {
      mockApi.get.mockReturnValue(of({
        categories: [{ name: 'portrait' }, { name: 'landscape' }, { name: 'macro' }],
      }));

      await component.loadCategories();

      expect(mockApi.get).toHaveBeenCalledWith('/comparison/category_weights');
      expect(component.categories()).toEqual(['portrait', 'landscape', 'macro']);
    });

    it('should auto-select first category', async () => {
      mockApi.get
        .mockReturnValueOnce(of({ categories: [{ name: 'portrait' }, { name: 'landscape' }] }))
        .mockReturnValueOnce(of({ weights: { aesthetic_percent: 35 }, modifiers: {} }))  // loadWeights
        .mockReturnValueOnce(of({ snapshots: [] }))  // loadSnapshots
        .mockReturnValue(of({}));  // loadWeightImpact etc

      await component.loadCategories();

      expect(component.selectedCategory()).toBe('portrait');
    });

    it('should not auto-select if categories are empty', async () => {
      mockApi.get.mockReturnValue(of({ categories: [] }));

      await component.loadCategories();

      expect(component.selectedCategory()).toBe('');
    });

    it('should show error on failure', async () => {
      mockApi.get.mockReturnValue(throwError(() => new Error('fail')));

      await component.loadCategories();

      expect(mockSnackBar.open).toHaveBeenCalledWith(
        'comparison.error_loading_categories',
        '',
        { duration: 4000 },
      );
    });
  });

  describe('selectCategory', () => {
    it('should set selected category and load weights and snapshots', async () => {
      mockApi.get.mockImplementation((path: string, params?: Record<string, unknown>) => {
        if (path === '/comparison/category_weights' && params?.['category'] === 'landscape') {
          return of({ weights: { aesthetic_percent: 35, composition_percent: 25 }, modifiers: { bonus: 0.5 } });
        }
        if (path === '/config/weight_snapshots') return of({ snapshots: [] });
        if (path === '/stats/categories/correlations') return of({});
        return of({});
      });

      await component.selectCategory('landscape');

      expect(component.selectedCategory()).toBe('landscape');
      expect(component.weights()).toEqual({ aesthetic_percent: 35, composition_percent: 25 });
      expect(component.modifiers()).toEqual({ bonus: 0.5 });
    });
  });

  describe('loadWeights', () => {
    it('should do nothing if no category is selected', async () => {
      component.selectedCategory.set('');
      mockApi.get.mockClear();

      await component.loadWeights();

      expect(mockApi.get).not.toHaveBeenCalled();
    });

    it('should set loading to false after completion', async () => {
      component.selectedCategory.set('portrait');
      mockApi.get.mockReturnValue(of({ weights: {}, modifiers: {} }));

      await component.loadWeights();

      expect(component.loading()).toBe(false);
    });
  });

  describe('saveWeights', () => {
    it('should post weights for selected category', async () => {
      component.selectedCategory.set('portrait');
      component.weights.set({ aesthetic_percent: 100 });

      await component.saveWeights();

      expect(mockApi.post).toHaveBeenCalledWith('/config/update_weights', {
        category: 'portrait',
        weights: { aesthetic_percent: 100 },
      });
    });

    it('should show success snackbar', async () => {
      component.selectedCategory.set('portrait');
      component.weights.set({ aesthetic_percent: 100 });

      await component.saveWeights();

      expect(mockSnackBar.open).toHaveBeenCalledWith('comparison.weights_saved', '', { duration: 3000 });
    });

    it('should do nothing if no category selected', async () => {
      component.selectedCategory.set('');

      await component.saveWeights();

      expect(mockApi.post).not.toHaveBeenCalled();
    });

    it('should set saving to false after completion', async () => {
      component.selectedCategory.set('portrait');
      component.weights.set({ aesthetic_percent: 100 });

      await component.saveWeights();

      expect(component.saving()).toBe(false);
    });
  });

  describe('loadPreview', () => {
    it('should fetch preview photos for selected category', async () => {
      component.selectedCategory.set('portrait');
      const photos = [
        { path: '/a.jpg', filename: 'a.jpg', aggregate: 8, aesthetic: 7, comp_score: 6, face_quality: 9 },
      ];
      mockApi.get.mockReturnValue(of({ photos }));

      await component.loadPreview();

      expect(mockApi.get).toHaveBeenCalledWith('/photos', expect.objectContaining({
        category: 'portrait',
        sort: 'aggregate',
      }));
      expect(component.previewPhotos()).toEqual(photos);
    });

    it('should set previewLoading to false after completion', async () => {
      component.selectedCategory.set('portrait');
      mockApi.get.mockReturnValue(of({ photos: [] }));

      await component.loadPreview();

      expect(component.previewLoading()).toBe(false);
    });
  });

  describe('snapshot CRUD', () => {
    it('should load snapshots', async () => {
      const snaps = [
        { id: 1, name: 'Baseline', category: 'portrait', weights: {}, created_at: '2024-01-01' },
      ];
      mockApi.get.mockReturnValue(of({ snapshots: snaps }));

      await component.loadSnapshots();

      expect(mockApi.get).toHaveBeenCalledWith('/config/weight_snapshots', expect.any(Object));
      expect(component.snapshots()).toEqual(snaps);
    });

    it('should save snapshot and reload list', async () => {
      component.selectedCategory.set('portrait');
      component.weights.set({ aesthetic_percent: 35 });
      component.snapshotName.set('My Snapshot');
      mockApi.post.mockReturnValue(of({}));
      mockApi.get.mockReturnValue(of({ snapshots: [] }));

      await component.saveSnapshot();

      expect(mockApi.post).toHaveBeenCalledWith('/config/save_snapshot', {
        category: 'portrait',
        description: 'My Snapshot',
      });
      expect(component.snapshotName()).toBe('');
    });

    it('should not save snapshot with empty name', async () => {
      component.snapshotName.set('   ');

      await component.saveSnapshot();

      expect(mockApi.post).not.toHaveBeenCalled();
    });

    it('should restore snapshot and reload weights', async () => {
      component.selectedCategory.set('portrait');
      mockApi.post.mockReturnValue(of({}));
      mockApi.get.mockReturnValue(of({ weights: { aesthetic_percent: 40 }, modifiers: {} }));

      await component.restoreSnapshot(5);

      expect(mockApi.post).toHaveBeenCalledWith('/config/restore_weights', { snapshot_id: 5 });
    });

    it('should show info snackbar for delete (not supported)', async () => {
      await component.deleteSnapshot(5);

      expect(mockSnackBar.open).toHaveBeenCalledWith('comparison.delete_not_supported', '', { duration: 3000 });
    });
  });

  describe('constructor', () => {
    it('should call loadCategories on construction', () => {
      expect(mockApi.get).toHaveBeenCalledWith('/comparison/category_weights');
    });
  });
});
