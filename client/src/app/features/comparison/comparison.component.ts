import { Component, inject, signal, computed, viewChild, ElementRef, effect, Pipe, PipeTransform, DestroyRef } from '@angular/core';
import { toObservable, takeUntilDestroyed } from '@angular/core/rxjs-interop';
import { FormsModule } from '@angular/forms';
import { MatCardModule } from '@angular/material/card';
import { MatTabsModule } from '@angular/material/tabs';
import { MatSliderModule } from '@angular/material/slider';
import { MatButtonModule } from '@angular/material/button';
import { MatSelectModule } from '@angular/material/select';
import { MatIconModule } from '@angular/material/icon';
import { MatSnackBar, MatSnackBarModule } from '@angular/material/snack-bar';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatTooltipModule } from '@angular/material/tooltip';
import { firstValueFrom, debounceTime, skip } from 'rxjs';
import { Chart, registerables } from 'chart.js';
import { ApiService } from '../../core/services/api.service';
import { AuthService } from '../../core/services/auth.service';
import { I18nService } from '../../core/services/i18n.service';
import { TranslatePipe } from '../../shared/pipes/translate.pipe';
import { FixedPipe } from '../../shared/pipes/fixed.pipe';
import { ThumbnailUrlPipe } from '../../shared/pipes/thumbnail-url.pipe';

Chart.register(...registerables);
Chart.defaults.color = '#a3a3a3';
Chart.defaults.borderColor = '#262626';

interface CategoryWeights {
  weights: Record<string, number>;
  modifiers: Record<string, number | boolean | string>;
}

interface PreviewPhoto {
  path: string;
  filename: string;
  aggregate: number;
  aesthetic: number;
  comp_score: number;
  face_quality: number;
  new_score?: number;
}

interface Snapshot {
  id: number;
  description: string;
  category: string;
  weights: Record<string, number>;
  timestamp: string;
}

interface WeightImpactResponse {
  correlations: Record<string, Record<string, number>>;
  configured_weights: Record<string, Record<string, number>>;
  dimensions: string[];
}

interface PairResponse {
  a?: string;
  b?: string;
  score_a?: number;
  score_b?: number;
  error?: string;
}

interface ComparisonStats {
  total_comparisons: number;
  winner_breakdown: Record<string, number>;
  category_breakdown: {category: string; count: number}[];
  unique_photos_compared: number;
  photos_with_learned_scores: number;
}

interface LearnedWeightsResponse {
  available: boolean;
  message?: string;
  current_weights?: Record<string, number>;
  suggested_weights?: Record<string, number>;
  accuracy_before?: number;
  accuracy_after?: number;
  improvement?: number;
  suggest_changes?: boolean;
  comparisons_used?: number;
  ties_included?: number;
  mispredicted_count?: number;
  category?: string;
}

const WEIGHT_ICONS: Record<string, string> = {
  aesthetic_percent: 'auto_awesome',
  composition_percent: 'grid_on',
  face_quality_percent: 'face',
  face_sharpness_percent: 'face_retouching_natural',
  eye_sharpness_percent: 'visibility',
  tech_sharpness_percent: 'center_focus_strong',
  exposure_percent: 'exposure',
  color_percent: 'palette',
  quality_percent: 'high_quality',
  contrast_percent: 'contrast',
  dynamic_range_percent: 'hdr_strong',
  saturation_percent: 'water_drop',
  noise_percent: 'grain',
  isolation_percent: 'filter_center_focus',
  power_point_percent: 'my_location',
  leading_lines_percent: 'timeline',
};

@Pipe({ name: 'weightIcon', standalone: true, pure: true })
export class WeightIconPipe implements PipeTransform {
  transform(key: string): string {
    return WEIGHT_ICONS[key] ?? 'tune';
  }
}

@Pipe({ name: 'weightLabelKey', standalone: true, pure: true })
export class WeightLabelKeyPipe implements PipeTransform {
  transform(key: string): string {
    return 'comparison.dim.' + key.replace('_percent', '');
  }
}

@Component({
  selector: 'app-comparison',
  imports: [
    FormsModule,
    MatCardModule,
    MatTabsModule,
    MatSliderModule,
    MatButtonModule,
    MatSelectModule,
    MatIconModule,
    MatSnackBarModule,
    MatFormFieldModule,
    MatInputModule,
    MatProgressSpinnerModule,
    MatTooltipModule,
    TranslatePipe,
    FixedPipe,
    ThumbnailUrlPipe,
    WeightIconPipe,
    WeightLabelKeyPipe,
  ],
  host: {
    '(window:keydown)': 'onKeydown($event)',
  },
  template: `
    <div class="p-4 md:p-6 max-w-7xl mx-auto">
      <h1 class="text-2xl font-semibold mb-6 flex items-center gap-2">
        <mat-icon>tune</mat-icon>
        {{ 'comparison.title' | translate }}
      </h1>

      <!-- Top bar: Category selector + Action buttons -->
      <div class="flex flex-wrap items-center gap-3 mb-6">
        <mat-form-field class="w-full md:w-72" subscriptSizing="dynamic">
          <mat-label>{{ 'comparison.category' | translate }}</mat-label>
          <mat-select
            [value]="selectedCategory()"
            (selectionChange)="selectCategory($event.value)">
            @for (cat of categories(); track cat) {
              <mat-option [value]="cat">{{ ('category_names.' + cat) | translate }}</mat-option>
            }
          </mat-select>
        </mat-form-field>
        <div class="flex gap-2 ml-auto flex-wrap">
          <button
            mat-flat-button
            [disabled]="!hasChanges() || !auth.isEdition() || saving()"
            (click)="saveWeights()"
            [matTooltip]="'comparison.save_tooltip' | translate">
            <mat-icon>save</mat-icon>
            {{ 'comparison.save' | translate }}
          </button>
          <button mat-stroked-button (click)="loadWeights()"
            [matTooltip]="'comparison.reset_tooltip' | translate">
            <mat-icon>refresh</mat-icon>
            {{ 'comparison.reset' | translate }}
          </button>
          <button
            mat-stroked-button
            [disabled]="hasChanges() || !auth.isEdition() || recalculating()"
            (click)="recalculateScores()"
            [matTooltip]="'comparison.recalculate_tooltip' | translate">
            @if (recalculating()) {
              <mat-spinner diameter="16" />
            } @else {
              <mat-icon>calculate</mat-icon>
            }
            {{ 'comparison.recalculate' | translate }}
          </button>
        </div>
      </div>

      @if (selectedCategory()) {
        <mat-tab-group class="mb-6" [selectedIndex]="0" (selectedIndexChange)="onTabChange($event)">
          <!-- Weights tab -->
          <mat-tab>
            <ng-template mat-tab-label>
              <mat-icon class="mr-2">sliders</mat-icon>
              {{ 'comparison.weights' | translate }}
            </ng-template>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-4">
              <!-- Left: Weight sliders -->
              <mat-card>
                <mat-card-header class="!flex !items-start">
                  <div class="flex-1">
                    <mat-card-title>{{ 'comparison.weight_sliders' | translate }}</mat-card-title>
                    <mat-card-subtitle>
                      {{ 'comparison.total' | translate }}: {{ weightTotal() }}%
                      @if (weightTotal() !== 100) {
                        <span class="text-amber-400">
                          ({{ 'comparison.should_be_100' | translate }})
                        </span>
                      }
                    </mat-card-subtitle>
                  </div>
                  @if (weightTotal() !== 100) {
                    <button mat-icon-button class="!w-8 !h-8 shrink-0" (click)="normalizeWeights()"
                      [matTooltip]="'stats.categories.weights.normalize' | translate">
                      <mat-icon>balance</mat-icon>
                    </button>
                  }
                </mat-card-header>
                <mat-card-content class="!pt-4">
                  @if (loading()) {
                    <div class="flex justify-center py-8">
                      <mat-spinner diameter="40" />
                    </div>
                  } @else {
                    <div class="flex flex-col gap-4">
                      @for (key of weightKeys(); track key) {
                          <div class="flex items-center gap-3">
                            <mat-icon class="text-gray-400 shrink-0">{{ key | weightIcon }}</mat-icon>
                            <span class="w-40 shrink-0 text-sm">
                              {{ key | weightLabelKey | translate }}
                            </span>
                            <mat-slider
                              class="grow"
                              [min]="0"
                              [max]="100"
                              [step]="1"
                              [discrete]="true"
                              [showTickMarks]="false">
                              <input
                                matSliderThumb
                                [value]="weights()[key]"
                                (valueChange)="setWeight(key, $event)" />
                            </mat-slider>
                            <span class="w-12 text-right text-sm font-mono tabular-nums">
                              {{ weights()[key] }}%
                            </span>
                          </div>
                      }
                    </div>
                  }
                </mat-card-content>
              </mat-card>

              <!-- Right: Weight Impact chart + Preview -->
              <div class="flex flex-col gap-6">
                <mat-card>
                  <mat-card-header>
                    <mat-card-title>{{ 'stats.weight_impact.title' | translate }}</mat-card-title>
                    <mat-card-subtitle>{{ 'stats.weight_impact.description' | translate }}</mat-card-subtitle>
                  </mat-card-header>
                  <mat-card-content class="!pt-4">
                    @if (weightImpactLoading()) {
                      <div class="flex justify-center py-8"><mat-spinner diameter="32" /></div>
                    } @else if (weightImpactData()) {
                      <div class="h-80">
                        <canvas #weightImpactCanvas></canvas>
                      </div>
                    } @else {
                      <p class="text-sm text-gray-400">{{ 'stats.weight_impact.empty' | translate }}</p>
                    }
                  </mat-card-content>
                </mat-card>

                <!-- Thumbnail preview -->
                <mat-card>
                  <mat-card-header>
                    <mat-card-title class="flex items-center gap-2">
                      {{ 'comparison.preview' | translate }}
                      @if (previewLoading()) {
                        <mat-spinner diameter="18" />
                      }
                    </mat-card-title>
                    <mat-card-subtitle>
                      {{ 'comparison.top_n_photos' | translate:{ count: previewCount } }}
                    </mat-card-subtitle>
                  </mat-card-header>
                  <mat-card-content class="!pt-4">
                    @if (previewPhotos().length > 0) {
                      <div class="grid grid-cols-2 md:grid-cols-3 gap-3">
                        @for (photo of previewPhotos(); track photo.path; let i = $index) {
                          <div class="relative rounded-lg overflow-hidden bg-neutral-900">
                            <img
                              [src]="photo.path | thumbnailUrl:320"
                              [alt]="photo.filename"
                              class="w-full object-contain bg-neutral-900" />
                            <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-2 py-1.5">
                              <span class="text-xs font-mono text-white">
                                #{{ i + 1 }}
                              </span>
                              <span class="text-xs font-mono text-gray-300 ml-2">
                                {{ (photo.new_score ?? photo.aggregate) | fixed:1 }}
                              </span>
                            </div>
                          </div>
                        }
                      </div>
                    } @else if (!previewLoading()) {
                      <p class="text-gray-500 text-sm">
                        {{ 'comparison.no_preview' | translate }}
                      </p>
                    }
                  </mat-card-content>
                </mat-card>
              </div>
            </div>
          </mat-tab>

          <!-- Snapshots tab -->
          <mat-tab>
            <ng-template mat-tab-label>
              <mat-icon class="mr-2">bookmark</mat-icon>
              {{ 'comparison.snapshots' | translate }}
            </ng-template>

            <p class="text-sm text-gray-400 mt-4 mb-4">
              {{ 'comparison.snapshots_description' | translate }}
            </p>

            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <!-- Save new snapshot -->
              <mat-card>
                <mat-card-header>
                  <mat-card-title class="!text-lg">{{ 'comparison.save_snapshot' | translate }}</mat-card-title>
                </mat-card-header>
                <mat-card-content class="!pt-4">
                  <mat-form-field class="w-full">
                    <mat-label>{{ 'comparison.snapshot_name' | translate }}</mat-label>
                    <input
                      matInput
                      [ngModel]="snapshotName()"
                      (ngModelChange)="snapshotName.set($event)"
                      (keyup.enter)="saveSnapshot()" />
                  </mat-form-field>
                  <button
                    mat-flat-button
                    [disabled]="!snapshotName().trim() || !auth.isEdition()"
                    (click)="saveSnapshot()">
                    <mat-icon>save</mat-icon>
                    {{ 'comparison.save' | translate }}
                  </button>
                </mat-card-content>
              </mat-card>

              <!-- Snapshot list -->
              <mat-card>
                <mat-card-header>
                  <mat-card-title class="!text-lg">{{ 'comparison.saved_snapshots' | translate }}</mat-card-title>
                </mat-card-header>
                <mat-card-content class="!pt-4">
                  @if (snapshots().length > 0) {
                    <div class="flex flex-col gap-2">
                      @for (snap of snapshots(); track snap.id) {
                        <div class="flex items-center justify-between gap-2 p-2 rounded bg-neutral-800/50">
                          <div class="min-w-0">
                            <div class="text-sm truncate">{{ snap.description }}</div>
                            <div class="text-xs text-gray-400">{{ snap.category }} &mdash; {{ snap.timestamp }}</div>
                          </div>
                          <div class="flex gap-1 shrink-0">
                            <button
                              mat-icon-button
                              [disabled]="!auth.isEdition()"
                              (click)="restoreSnapshot(snap.id)"
                              [attr.aria-label]="'comparison.restore' | translate">
                              <mat-icon>restore</mat-icon>
                            </button>
                            <button
                              mat-icon-button
                              [disabled]="!auth.isEdition()"
                              (click)="deleteSnapshot(snap.id)"
                              [attr.aria-label]="'comparison.delete' | translate">
                              <mat-icon>delete</mat-icon>
                            </button>
                          </div>
                        </div>
                      }
                    </div>
                  } @else {
                    <p class="text-gray-500 text-sm">
                      {{ 'comparison.no_snapshots' | translate }}
                    </p>
                  }
                </mat-card-content>
              </mat-card>
            </div>
          </mat-tab>

          <!-- A/B Compare tab -->
          <mat-tab>
            <ng-template mat-tab-label>
              <mat-icon class="mr-2">compare</mat-icon>
              {{ 'comparison.compare_tab' | translate }}
            </ng-template>

            <div class="mt-4">
              <!-- Strategy selector + keyboard hints -->
              <div class="flex flex-wrap items-center gap-3 mb-4">
                <mat-form-field class="w-56" subscriptSizing="dynamic">
                  <mat-label>{{ 'compare.strategy' | translate }}</mat-label>
                  <mat-select [value]="strategy()" (selectionChange)="onStrategyChange($event.value)">
                    @for (s of strategies; track s) {
                      <mat-option [value]="s">{{ ('compare.strategies.' + s) | translate }}</mat-option>
                    }
                  </mat-select>
                </mat-form-field>
                <button mat-icon-button (click)="showStrategyHelp.set(!showStrategyHelp())"
                  [matTooltip]="'compare.tooltips.strategy_info' | translate">
                  <mat-icon>help_outline</mat-icon>
                </button>
                <span class="ml-auto text-xs text-gray-500 hidden md:inline">
                  {{ 'compare.keyboard.hint' | translate }}
                  <kbd class="px-1 rounded bg-neutral-700 text-gray-300">&#8592;</kbd> {{ 'compare.keyboard.left_wins' | translate }} ·
                  <kbd class="px-1 rounded bg-neutral-700 text-gray-300">&#8594;</kbd> {{ 'compare.keyboard.right_wins' | translate }} ·
                  <kbd class="px-1 rounded bg-neutral-700 text-gray-300">T</kbd> {{ 'compare.keyboard.equal' | translate }} ·
                  <kbd class="px-1 rounded bg-neutral-700 text-gray-300">S</kbd> {{ 'compare.keyboard.skip' | translate }}
                </span>
              </div>

              <!-- Strategy help panel -->
              @if (showStrategyHelp()) {
                <mat-card class="mb-4">
                  <mat-card-content class="!py-3">
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                      @for (s of strategies; track s) {
                        <div>
                          <div class="font-medium">{{ ('compare.strategy_help.' + s + '_title') | translate }}</div>
                          <div class="text-gray-400 text-xs">{{ ('compare.strategy_help.' + s + '_desc') | translate }}</div>
                        </div>
                      }
                    </div>
                  </mat-card-content>
                </mat-card>
              }

              <!-- Photos + stats sidebar -->
              <div class="grid grid-cols-1 lg:grid-cols-[1fr_260px] gap-4">
                <!-- Photo pair -->
                <mat-card>
                  <mat-card-content class="!pt-4">
                    @if (pairError()) {
                      <div class="text-red-400 text-sm mb-4">{{ pairError() }}</div>
                    }

                    @if (!pairA() && !pairLoading()) {
                      <div class="flex flex-col items-center py-8 gap-4">
                        <p class="text-sm text-gray-400">{{ 'comparison.compare_description' | translate }}</p>
                        <button mat-flat-button (click)="loadNextPair()">
                          <mat-icon>play_arrow</mat-icon>
                          {{ 'comparison.start_comparing' | translate }}
                        </button>
                      </div>
                    } @else if (pairLoading()) {
                      <div class="flex justify-center py-8">
                        <mat-spinner diameter="40" />
                      </div>
                    } @else if (pairA() && pairB()) {
                      <div class="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                        <button
                          class="relative rounded-lg overflow-hidden bg-neutral-900 cursor-pointer border-2 border-transparent hover:border-green-500 transition-colors text-left p-0"
                          [disabled]="pairSubmitting()"
                          (click)="submitComparison('a')">
                          <img
                            [src]="pairA()! | thumbnailUrl:640"
                            alt="Photo A"
                            class="w-full max-h-[60vh] object-contain" />
                          <div class="absolute top-2 left-2 text-xs font-mono bg-black/60 px-2 py-0.5 rounded">A</div>
                          <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-3 py-2">
                            <span class="text-sm font-mono text-white">{{ pairScoreA() | fixed:1 }}</span>
                          </div>
                        </button>
                        <button
                          class="relative rounded-lg overflow-hidden bg-neutral-900 cursor-pointer border-2 border-transparent hover:border-green-500 transition-colors text-left p-0"
                          [disabled]="pairSubmitting()"
                          (click)="submitComparison('b')">
                          <img
                            [src]="pairB()! | thumbnailUrl:640"
                            alt="Photo B"
                            class="w-full max-h-[60vh] object-contain" />
                          <div class="absolute top-2 right-2 text-xs font-mono bg-black/60 px-2 py-0.5 rounded">B</div>
                          <div class="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 to-transparent px-3 py-2">
                            <span class="text-sm font-mono text-white">{{ pairScoreB() | fixed:1 }}</span>
                          </div>
                        </button>
                      </div>
                      <div class="flex justify-center gap-3">
                        <button mat-stroked-button [disabled]="pairSubmitting()" (click)="submitComparison('tie')">
                          <mat-icon>drag_handle</mat-icon>
                          {{ 'comparison.tie' | translate }}
                        </button>
                        <button mat-stroked-button [disabled]="pairSubmitting()" (click)="loadNextPair()">
                          <mat-icon>skip_next</mat-icon>
                          {{ 'comparison.skip' | translate }}
                        </button>
                        @if (comparisonCount() > 0) {
                          <span class="flex items-center text-sm text-gray-400 ml-2">
                            {{ comparisonCount() }} {{ 'comparison.comparisons_completed' | translate }}
                          </span>
                        }
                      </div>
                    } @else {
                      <div class="text-center py-8 text-gray-400">
                        {{ 'comparison.no_more_pairs' | translate }}
                      </div>
                    }
                  </mat-card-content>
                </mat-card>

                <!-- Sidebar: Stats + Weight suggestions -->
                <div class="flex flex-col gap-4">
                  <!-- Stats -->
                  <mat-card>
                    <mat-card-header>
                      <mat-card-title class="!text-sm">{{ 'compare.stats.title' | translate }}</mat-card-title>
                    </mat-card-header>
                    <mat-card-content class="!pt-2">
                      @if (comparisonStats(); as stats) {
                        <div class="grid grid-cols-2 gap-x-3 gap-y-1 text-sm">
                          <span class="text-gray-400">{{ 'compare.stats.total_comparisons' | translate }}</span>
                          <span class="text-right font-mono">{{ stats.total_comparisons }}</span>
                          <span class="text-gray-400">{{ 'compare.stats.a_wins' | translate }}</span>
                          <span class="text-right font-mono">{{ stats.winner_breakdown['a'] || 0 }}</span>
                          <span class="text-gray-400">{{ 'compare.stats.b_wins' | translate }}</span>
                          <span class="text-right font-mono">{{ stats.winner_breakdown['b'] || 0 }}</span>
                          <span class="text-gray-400">{{ 'compare.stats.ties_label' | translate }}</span>
                          <span class="text-right font-mono">{{ stats.winner_breakdown['tie'] || 0 }}</span>
                        </div>
                      } @else {
                        <p class="text-xs text-gray-500">{{ 'comparison.no_data_yet' | translate }}</p>
                      }
                    </mat-card-content>
                  </mat-card>

                  <!-- Weight Suggestions -->
                  <mat-card>
                    <mat-card-header>
                      <mat-card-title class="!text-sm">{{ 'compare.actions.suggest_weights' | translate }}</mat-card-title>
                    </mat-card-header>
                    <mat-card-content class="!pt-2">
                      <button mat-stroked-button class="w-full mb-3" [disabled]="learnedWeightsLoading()"
                        (click)="loadLearnedWeights()">
                        @if (learnedWeightsLoading()) {
                          <mat-spinner diameter="16" />
                        }
                        {{ 'compare.actions.suggest_weights' | translate }}
                      </button>
                      @if (learnedWeights(); as lw) {
                        @if (lw.available) {
                          <div class="text-xs space-y-2">
                            <div class="text-gray-400">
                              {{ 'compare.weights.learned_from' | translate:{ count: lw.comparisons_used ?? 0 } }}
                              @if (lw.ties_included) {
                                ({{ 'compare.weights.incl_ties' | translate:{ count: lw.ties_included } }})
                              }
                            </div>
                            <div class="flex items-center gap-2">
                              <span class="text-gray-400">{{ 'compare.weights.prediction_accuracy' | translate }}:</span>
                              <span class="font-mono">{{ (lw.accuracy_before ?? 0) | fixed:0 }}%</span>
                              <span class="text-gray-500">&rarr;</span>
                              <span class="font-mono text-green-400">{{ (lw.accuracy_after ?? 0) | fixed:0 }}%</span>
                            </div>
                            @if (lw.mispredicted_count) {
                              <div class="text-gray-500">
                                {{ 'compare.weights.mispredicted' | translate:{ count: lw.mispredicted_count } }}
                              </div>
                            }
                            @if (lw.suggest_changes && lw.suggested_weights) {
                              <div class="space-y-0.5 mb-3">
                                @for (key of weightKeys(); track key) {
                                  <div class="flex items-center text-xs">
                                    <span class="w-28 shrink-0 truncate text-gray-400">{{ key | weightLabelKey | translate }}</span>
                                    <span class="font-mono w-10 shrink-0 text-right tabular-nums">{{ weights()[key] || 0 }}</span>
                                    <span class="w-6 shrink-0 text-center text-gray-500">&rarr;</span>
                                    <span class="font-mono w-10 shrink-0 text-right tabular-nums text-green-400">{{ lw.suggested_weights[key] || 0 }}</span>
                                  </div>
                                }
                              </div>
                              <button mat-flat-button class="w-full"
                                [disabled]="!auth.isEdition()"
                                (click)="applySuggestedWeights()">
                                <mat-icon>auto_fix_high</mat-icon>
                                {{ 'comparison.apply_suggested' | translate }}
                              </button>
                            } @else {
                              <p class="text-amber-400">{{ 'compare.weights.already_good' | translate }}</p>
                            }
                          </div>
                        } @else {
                          <p class="text-xs text-gray-500">{{ lw.message }}</p>
                        }
                      }
                    </mat-card-content>
                  </mat-card>
                </div>
              </div>
            </div>
          </mat-tab>
        </mat-tab-group>
      }
    </div>
  `,
})
export class ComparisonComponent {
  private api = inject(ApiService);
  private snackBar = inject(MatSnackBar);
  private i18n = inject(I18nService);
  private destroyRef = inject(DestroyRef);
  auth = inject(AuthService);

  readonly previewCount = 6;
  private charts = new Map<string, Chart>();

  categories = signal<string[]>([]);
  selectedCategory = signal<string>('');
  weights = signal<Record<string, number>>({});
  private savedWeights = signal<Record<string, number>>({});
  modifiers = signal<Record<string, number | boolean | string>>({});
  loading = signal(false);
  saving = signal(false);
  recalculating = signal(false);

  hasChanges = computed(() => {
    const current = this.weights();
    const saved = this.savedWeights();
    return Object.keys(current).some(k => current[k] !== saved[k]);
  });

  previewPhotos = signal<PreviewPhoto[]>([]);
  previewLoading = signal(false);

  snapshots = signal<Snapshot[]>([]);
  snapshotName = signal('');

  // Weight impact chart
  weightImpactData = signal<WeightImpactResponse | null>(null);
  weightImpactLoading = signal(false);
  weightImpactCanvas = viewChild<ElementRef<HTMLCanvasElement>>('weightImpactCanvas');

  // A/B Comparison
  pairA = signal<string | null>(null);
  pairB = signal<string | null>(null);
  pairScoreA = signal(0);
  pairScoreB = signal(0);
  pairLoading = signal(false);
  pairSubmitting = signal(false);
  pairError = signal<string | null>(null);
  comparisonCount = signal(0);
  strategy = signal<string>('uncertainty');
  readonly strategies = ['uncertainty', 'boundary', 'active', 'random'] as const;
  comparisonStats = signal<ComparisonStats | null>(null);
  learnedWeights = signal<LearnedWeightsResponse | null>(null);
  learnedWeightsLoading = signal(false);
  showStrategyHelp = signal(false);

  /** Derive weight keys dynamically from whatever the API returns */
  weightKeys = computed(() => Object.keys(this.weights()).filter(k => k.endsWith('_percent')));

  weightTotal = computed(() => {
    const w = this.weights();
    return Object.values(w).reduce((sum, v) => sum + (v || 0), 0);
  });

  constructor() {
    this.loadCategories();

    // Weight impact chart effect
    effect(() => {
      const data = this.weightImpactData();
      const cat = this.selectedCategory();
      if (data && cat) {
        this.buildWeightImpactChart(data, cat);
      }
    });

    // Debounced auto-refresh preview on weight change
    toObservable(this.weights).pipe(
      skip(1),
      debounceTime(600),
      takeUntilDestroyed(),
    ).subscribe(() => {
      if (this.selectedCategory()) this.loadPreview();
    });

    // Destroy Chart.js instances on component teardown
    this.destroyRef.onDestroy(() => {
      this.charts.forEach(chart => chart.destroy());
      this.charts.clear();
    });
  }

  async loadCategories(): Promise<void> {
    try {
      const res = await firstValueFrom(this.api.get<{categories: {name: string}[]}>('/comparison/category_weights'));
      const cats = (res.categories ?? []).map(c => c.name);
      this.categories.set(cats);
      if (cats.length > 0 && !this.selectedCategory()) {
        this.selectCategory(cats[0]);
      }
    } catch {
      this.showError('comparison.error_loading_categories');
    }
  }

  async selectCategory(category: string): Promise<void> {
    this.selectedCategory.set(category);
    await Promise.all([this.loadWeights(), this.loadSnapshots(), this.loadWeightImpact()]);
  }

  async loadWeights(): Promise<void> {
    const cat = this.selectedCategory();
    if (!cat) return;

    this.loading.set(true);
    try {
      const data = await firstValueFrom(
        this.api.get<CategoryWeights>('/comparison/category_weights', {category: cat}),
      );
      this.weights.set({ ...data.weights });
      this.savedWeights.set({ ...data.weights });
      this.modifiers.set({ ...data.modifiers });
    } catch {
      this.showError('comparison.error_loading_weights');
    } finally {
      this.loading.set(false);
    }
  }

  setWeight(key: string, value: number): void {
    this.weights.update(w => ({ ...w, [key]: value }));
  }

  async saveWeights(): Promise<void> {
    const cat = this.selectedCategory();
    if (!cat) return;

    // Normalize to 100% before saving
    if (this.weightTotal() !== 100) this.normalizeWeights();

    this.saving.set(true);
    try {
      await firstValueFrom(
        this.api.post('/config/update_weights', {
          category: cat,
          weights: this.weights(),
        }),
      );
      this.savedWeights.set({ ...this.weights() });
      this.snackBar.open(this.i18n.t('comparison.weights_saved'), '', { duration: 3000 });
    } catch {
      this.showError('comparison.error_saving_weights');
    } finally {
      this.saving.set(false);
    }
  }

  async recalculateScores(): Promise<void> {
    const cat = this.selectedCategory();
    if (!cat) return;

    this.recalculating.set(true);
    try {
      const result = await firstValueFrom(
        this.api.post<{ success: boolean; message?: string }>('/stats/categories/recompute', { category: cat }),
      );
      this.snackBar.open(result.message ?? this.i18n.t('comparison.recalculated'), '', { duration: 5000 });
      // Refresh preview photos and weight impact chart with new scores
      this.loadPreview();
      this.loadWeightImpact();
    } catch {
      this.showError('comparison.error_recalculating');
    } finally {
      this.recalculating.set(false);
    }
  }

  async loadPreview(): Promise<void> {
    const cat = this.selectedCategory();
    if (!cat) return;

    this.previewLoading.set(true);
    try {
      const data = await firstValueFrom(
        this.api.get<{photos: PreviewPhoto[]}>('/photos', {
          category: cat,
          sort: 'aggregate',
          sort_direction: 'DESC',
          per_page: this.previewCount,
          page: 1,
        }),
      );
      this.previewPhotos.set(data.photos ?? []);
    } catch {
      this.showError('comparison.error_loading_preview');
    } finally {
      this.previewLoading.set(false);
    }
  }

  async loadSnapshots(): Promise<void> {
    const cat = this.selectedCategory();
    try {
      const res = await firstValueFrom(this.api.get<{snapshots: Snapshot[]}>('/config/weight_snapshots', cat ? {category: cat} : {}));
      this.snapshots.set(res.snapshots ?? []);
    } catch {
      this.showError('comparison.error_loading_snapshots');
    }
  }

  async saveSnapshot(): Promise<void> {
    const name = this.snapshotName().trim();
    if (!name) return;

    try {
      await firstValueFrom(
        this.api.post('/config/save_snapshot', {
          category: this.selectedCategory(),
          description: name,
        }),
      );
      this.snapshotName.set('');
      this.snackBar.open(this.i18n.t('comparison.snapshot_saved'), '', { duration: 3000 });
      await this.loadSnapshots();
    } catch {
      this.showError('comparison.error_saving_snapshot');
    }
  }

  async restoreSnapshot(id: number): Promise<void> {
    try {
      await firstValueFrom(this.api.post('/config/restore_weights', {snapshot_id: id}));
      this.snackBar.open(this.i18n.t('comparison.snapshot_restored'), '', { duration: 3000 });
      await this.loadWeights();
    } catch {
      this.showError('comparison.error_restoring_snapshot');
    }
  }

  async deleteSnapshot(_id: number): Promise<void> {
    this.snackBar.open(this.i18n.t('comparison.delete_not_supported'), '', { duration: 3000 });
  }

  // --- Weight Impact ---

  async loadWeightImpact(): Promise<void> {
    this.weightImpactLoading.set(true);
    try {
      const data = await firstValueFrom(
        this.api.get<WeightImpactResponse>('/stats/categories/correlations'),
      );
      this.weightImpactData.set(data);
    } catch { /* empty */ }
    finally { this.weightImpactLoading.set(false); }
  }

  private buildWeightImpactChart(data: WeightImpactResponse, category: string): void {
    const ref = this.weightImpactCanvas();
    if (!ref) return;
    this.destroyChart('weightImpact');
    const ctx = ref.nativeElement.getContext('2d');
    if (!ctx) return;

    const weights = data.configured_weights?.[category] ?? {};
    const corrs = data.correlations?.[category] ?? {};

    // Use the same keys and order as the weight sliders
    const activeDims = this.weightKeys().map(k => k.replace('_percent', ''));
    if (activeDims.length === 0) return;

    const labels = activeDims.map((d: string) => this.i18n.t('stats.weight_impact.dims.' + d));
    const weightValues = activeDims.map((d: string) => weights[d] ?? 0);
    const corrValues = activeDims.map((d: string) => Math.abs(corrs[d] ?? 0) * 100);

    this.charts.set('weightImpact', new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          {
            label: this.i18n.t('stats.weight_impact.configured'),
            data: weightValues,
            backgroundColor: '#3b82f6cc',
            borderColor: '#3b82f6',
            borderWidth: 1,
            borderRadius: 3,
          },
          {
            label: this.i18n.t('stats.weight_impact.actual_impact'),
            data: corrValues,
            backgroundColor: '#22c55ecc',
            borderColor: '#22c55e',
            borderWidth: 1,
            borderRadius: 3,
          },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: true, labels: { color: '#d4d4d4', boxWidth: 12 } },
          tooltip: {
            callbacks: {
              label: (tooltipCtx) => {
                const val = tooltipCtx.parsed.x ?? 0;
                return `${tooltipCtx.dataset.label}: ${val.toFixed(1)}%`;
              },
            },
          },
        },
        scales: {
          x: {
            grid: { color: '#262626' },
            ticks: { color: '#a3a3a3', callback: (v) => v + '%' },
            max: 100,
          },
          y: { grid: { display: false }, ticks: { color: '#d4d4d4', font: { size: 11 } } },
        },
      },
    }));
  }

  // --- A/B Comparison ---

  onTabChange(index: number): void {
    // Auto-load pair when switching to Compare tab (index 2)
    if (index === 2 && !this.pairA() && !this.pairLoading()) {
      this.loadNextPair();
    }
  }

  onStrategyChange(value: string): void {
    this.strategy.set(value);
    // Reload pair if a comparison is in progress
    if (this.pairA()) this.loadNextPair();
  }

  async loadNextPair(): Promise<void> {
    const cat = this.selectedCategory();
    if (!cat) return;

    this.pairLoading.set(true);
    this.pairError.set(null);
    this.loadComparisonStats();
    try {
      const data = await firstValueFrom(
        this.api.get<PairResponse>('/comparison/next_pair', { category: cat, strategy: this.strategy() }),
      );
      if (data.error) {
        this.pairA.set(null);
        this.pairB.set(null);
        this.pairError.set(data.error);
      } else if (data.a && data.b) {
        this.pairA.set(data.a);
        this.pairB.set(data.b);
        this.pairScoreA.set(data.score_a ?? 0);
        this.pairScoreB.set(data.score_b ?? 0);
      } else {
        this.pairA.set(null);
        this.pairB.set(null);
        this.pairError.set(this.i18n.t('comparison.no_more_pairs'));
      }
    } catch {
      this.pairError.set(this.i18n.t('comparison.error_loading_pair'));
    } finally {
      this.pairLoading.set(false);
    }
  }

  async submitComparison(winner: 'a' | 'b' | 'tie'): Promise<void> {
    const a = this.pairA();
    const b = this.pairB();
    const cat = this.selectedCategory();
    if (!a || !b || !cat) return;

    this.pairSubmitting.set(true);
    try {
      await firstValueFrom(
        this.api.post('/comparison/submit', {
          photo_a: a,
          photo_b: b,
          winner,
          category: cat,
        }),
      );
      this.comparisonCount.update(c => c + 1);
      this.loadComparisonStats();
      await this.loadNextPair();
    } catch {
      this.pairError.set(this.i18n.t('comparison.error_submitting'));
    } finally {
      this.pairSubmitting.set(false);
    }
  }

  // --- Comparison Stats & Suggestions ---

  async loadComparisonStats(): Promise<void> {
    try {
      const data = await firstValueFrom(
        this.api.get<ComparisonStats>('/comparison/stats'),
      );
      this.comparisonStats.set(data);
    } catch { /* non-critical */ }
  }

  async loadLearnedWeights(): Promise<void> {
    const cat = this.selectedCategory();
    if (!cat) return;
    this.learnedWeightsLoading.set(true);
    try {
      const data = await firstValueFrom(
        this.api.get<LearnedWeightsResponse>('/comparison/learned_weights', { category: cat }),
      );
      this.learnedWeights.set(data);
    } catch {
      this.showError('comparison.error_loading_suggestions');
    } finally {
      this.learnedWeightsLoading.set(false);
    }
  }

  applySuggestedWeights(): void {
    const lw = this.learnedWeights();
    if (!lw?.suggested_weights) return;
    const current = this.weights();
    const merged = { ...current, ...lw.suggested_weights };
    this.weights.set(merged);
    this.snackBar.open(this.i18n.t('comparison.optimized'), '', { duration: 3000 });
  }

  normalizeWeights(): void {
    const w = this.weights();
    const total = Object.values(w).reduce((sum, v) => sum + (v || 0), 0);
    if (total === 0) return;
    const factor = 100 / total;
    const normalized: Record<string, number> = {};
    let runningTotal = 0;
    const keys = Object.keys(w);
    for (let i = 0; i < keys.length; i++) {
      if (i === keys.length - 1) {
        normalized[keys[i]] = 100 - runningTotal;
      } else {
        normalized[keys[i]] = Math.round(w[keys[i]] * factor);
        runningTotal += normalized[keys[i]];
      }
    }
    this.weights.set(normalized);
  }

  onKeydown(event: KeyboardEvent): void {
    const tag = (event.target as HTMLElement)?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (!this.pairA() || !this.pairB() || this.pairSubmitting() || this.pairLoading()) return;
    switch (event.key) {
      case 'ArrowLeft': this.submitComparison('a'); event.preventDefault(); break;
      case 'ArrowRight': this.submitComparison('b'); event.preventDefault(); break;
      case 't': case 'T': this.submitComparison('tie'); event.preventDefault(); break;
      case 's': case 'S': this.loadNextPair(); event.preventDefault(); break;
    }
  }

  // --- Helpers ---

  private destroyChart(id: string): void {
    const existing = this.charts.get(id);
    if (existing) {
      existing.destroy();
      this.charts.delete(id);
    }
  }

  private showError(key: string): void {
    this.snackBar.open(this.i18n.t(key), '', { duration: 4000 });
  }
}
